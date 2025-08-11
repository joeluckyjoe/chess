import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import chess
import chess.pgn
from pathlib import Path
import sys
from collections import deque
from tqdm import tqdm
import re
import gc
import argparse

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))

from config import get_paths, config_params
from gnn_agent.neural_network.temporal_model import TemporalPolicyValueModel
from gnn_agent.neural_network.policy_value_model import PolicyValueModel as EncoderPolicyValueModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from hardware_setup import get_device
from run_expert_sparring import GNN_METADATA, prepare_sequence_batch

# --- Configuration ---
EPOCHS = 3
BATCH_SIZE = config_params.get("BATCH_SIZE", 256)
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 8
SAMPLES_PER_BATCH = 50000 

def get_value_target(result: str, player_turn: chess.Color) -> float:
    if result == '1-0': return 1.0 if player_turn == chess.WHITE else -1.0
    elif result == '0-1': return -1.0 if player_turn == chess.WHITE else 1.0
    return 0.0

def process_pgn_stream_to_samples(pgn_stream, device, pbar):
    """
    Reads games from a PGN file stream and yields samples.
    """
    empty_board = chess.Board()
    initial_gnn, initial_cnn, _ = convert_to_gnn_input(empty_board, device)
    initial_state_tuple = (initial_gnn, initial_cnn)

    while True:
        try:
            game = chess.pgn.read_game(pgn_stream)
            if game is None: break
            if pbar: pbar.update(1)
            
            result = game.headers.get("Result", "*")
            if result == '*': continue

            board = game.board()
            state_deque = deque([initial_state_tuple] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
            
            for move in game.mainline_moves():
                player_turn = board.turn
                policy_target = torch.zeros(get_action_space_size())
                action_index = move_to_index(move, board)
                if action_index is None: continue
                policy_target[action_index] = 1.0
                
                value_target = torch.tensor([get_value_target(result, player_turn)], dtype=torch.float32)

                yield {
                    'state_sequence': list(state_deque),
                    'policy': policy_target,
                    'value_target': value_target
                }

                board.push(move)
                gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
                state_deque.append((gnn_data, cnn_tensor))
        except (ValueError, KeyError, IndexError):
            continue

# <<< MODIFIED: Validation function now streams data instead of pre-loading >>>
def validate(model, validation_pgn_files, device):
    """Runs an evaluation on the validation set by streaming it file by file."""
    model.eval()
    total_val_loss, correct_policy_preds, total_policy_samples = 0, 0, 0
    num_batches = 0

    print("Running validation...")
    for pgn_path in validation_pgn_files:
        with open(pgn_path, encoding='utf-8', errors='ignore') as pgn_stream:
            sample_generator = process_pgn_stream_to_samples(pgn_stream, device, None)
            
            while True:
                validation_samples = [next(sample_generator) for _ in range(SAMPLES_PER_BATCH) if (s := next(sample_generator, None)) is not None]
                if not validation_samples: break

                validation_loader = DataLoader(validation_samples, batch_size=BATCH_SIZE)
                
                with torch.no_grad():
                    for batch in validation_loader:
                        gnn_batch, cnn_batch, target_policies, target_values = prepare_sequence_batch(batch, device)
                        policy_logits, value_preds = model(gnn_batch, cnn_batch)
                        policy_loss = -(torch.nn.functional.log_softmax(policy_logits, dim=1) * target_policies).sum(dim=1).mean()
                        value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(-1), target_values.squeeze(-1))
                        total_val_loss += (policy_loss + value_loss).item()
                        
                        predicted_moves = torch.argmax(policy_logits, dim=1)
                        correct_moves = torch.argmax(target_policies, dim=1)
                        correct_policy_preds += (predicted_moves == correct_moves).sum().item()
                        total_policy_samples += len(target_policies)
                        num_batches += 1

    if num_batches == 0: return 0.0, 0.0
    return total_val_loss / num_batches, (correct_policy_preds / total_policy_samples) * 100

def main():
    parser = argparse.ArgumentParser(description="Iterative supervised training.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a checkpoint to resume training.')
    args = parser.parse_args()

    paths = get_paths()
    device = get_device()

    dummy_encoder = EncoderPolicyValueModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
        embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
    ).to(device)
    model = TemporalPolicyValueModel(encoder_model=dummy_encoder, policy_size=get_action_space_size(), d_model=config_params['EMBED_DIM']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_corpus_dir = paths.drive_project_root / 'pgn_corpus'
    validation_corpus_dir = paths.drive_project_root / 'validation_corpus'
    
    def get_part_num(f):
        match = re.search(r'(\d+)', f.name)
        return int(match.group(1)) if match else -1

    train_pgn_files = sorted(train_corpus_dir.glob('*.pgn'), key=get_part_num)
    validation_pgn_files = list(validation_corpus_dir.glob('*.pgn'))

    if not train_pgn_files or not validation_pgn_files:
        print("[FATAL] Both a training and validation corpus are required.")
        sys.exit(1)

    # <<< MODIFIED: Removed the pre-loading of the validation set >>>

    start_epoch = 0
    last_processed_file_index = -1

    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        last_processed_file_index = checkpoint.get('last_processed_file_index', -1)

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n--- Starting Epoch {epoch + 1}/{EPOCHS} ---")
        
        start_index = last_processed_file_index + 1
        last_processed_file_index = -1 

        for i in range(start_index, len(train_pgn_files)):
            pgn_path = train_pgn_files[i]
            print(f"\nProcessing training file: {pgn_path.name} ({i+1}/{len(train_pgn_files)})")
            
            with open(pgn_path, encoding='utf-8', errors='ignore') as pgn_stream:
                pbar = tqdm(desc=f"Reading {pgn_path.name}")
                sample_generator = process_pgn_stream_to_samples(pgn_stream, device, pbar)
                
                batch_num = 0
                is_file_done = False
                while not is_file_done:
                    train_samples = []
                    for _ in range(SAMPLES_PER_BATCH):
                        sample = next(sample_generator, None)
                        if sample is None:
                            is_file_done = True
                            break
                        train_samples.append(sample)

                    if not train_samples: break
                    batch_num += 1

                    train_loader = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True)
                    
                    model.train()
                    for batch in tqdm(train_loader, desc=f"Training on {pgn_path.name} batch {batch_num}", leave=False):
                        gnn_batch, cnn_batch, target_policies, target_values = prepare_sequence_batch(batch, device)
                        optimizer.zero_grad()
                        policy_logits, value_preds = model(gnn_batch, cnn_batch)
                        policy_loss = -(torch.nn.functional.log_softmax(policy_logits, dim=1) * target_policies).sum(dim=1).mean()
                        value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(-1), target_values.squeeze(-1))
                        total_loss = policy_loss + value_loss
                        total_loss.backward()
                        optimizer.step()
            
            checkpoint_path = paths.checkpoints_dir / f"iterative_supervised_checkpoint_epoch{epoch+1}_file{i+1}.pth.tar"
            torch.save({ 'epoch': epoch, 'last_processed_file_index': i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict() }, checkpoint_path)
            
            # <<< MODIFIED: Pass the list of files to the validation function >>>
            val_loss, val_accuracy = validate(model, validation_pgn_files, device)
            print(f"File {pgn_path.name} complete. | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
            print(f"Checkpoint saved to {checkpoint_path}")

            del train_samples, train_loader
            gc.collect()

if __name__ == "__main__":
    main()