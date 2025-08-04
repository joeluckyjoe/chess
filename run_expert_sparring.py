import torch
import torch.optim as optim
import argparse
import sys
import chess
import chess.pgn
import datetime
from stockfish import Stockfish
from pathlib import Path
import numpy as np
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from typing import Dict, List, TypedDict
from collections import deque

# --- Import from config ---
from config import get_paths, config_params

# --- Project-specific Imports ---
from gnn_agent.neural_network.policy_value_model import PolicyValueModel as EncoderPolicyValueModel
from gnn_agent.neural_network.temporal_model import TemporalPolicyValueModel
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.search.mcts import MCTS
from hardware_setup import get_device

# --- Constants ---
SEQUENCE_LENGTH = 8 # As defined in the Global Plan

# --- Type Hint for Game Memory ---
class GameStep(TypedDict):
    state_sequence: List[object]
    policy: torch.Tensor
    value_target: torch.Tensor

# --- Helper Functions ---
GNN_METADATA = (['square', 'piece'],[('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')])

def format_policy_for_training(policy_dict: Dict[chess.Move, float], board: chess.Board) -> torch.Tensor:
    policy_tensor = torch.zeros(get_action_space_size())
    for move, prob in policy_dict.items():
        try:
            action_index = move_to_index(move, board)
            if action_index is not None:
                policy_tensor[action_index] = prob
        except IndexError:
            print(f"Warning: IndexError for move {move} skipped in policy formatting.")
            continue
    return policy_tensor

def prepare_sequence_batch(batch_memory: List[GameStep], device: torch.device):
    batch_size = len(batch_memory)
    all_gnn_data = []
    cnn_sequences = []

    for step in batch_memory:
        gnn_sequence_for_step = [s[0] for s in step['state_sequence']]
        cnn_sequence_for_step = [s[1] for s in step['state_sequence']]

        all_gnn_data.extend(gnn_sequence_for_step)
        cnn_sequences.append(torch.stack(cnn_sequence_for_step))

    gnn_batch = Batch.from_data_list(all_gnn_data).to(device)
    cnn_batch = torch.stack(cnn_sequences).to(device)

    target_policies = torch.stack([item['policy'] for item in batch_memory]).to(device)
    target_values = torch.stack([item['value_target'] for item in batch_memory]).to(device)

    return gnn_batch, cnn_batch, target_policies, target_values


def train_on_game(model: TemporalPolicyValueModel, optimizer: optim.Optimizer, game_memory: List[GameStep], batch_size: int, device: torch.device):
    """Trains the temporal model on the data collected from a single game."""
    if not game_memory:
        return 0.0, 0.0

    model.train() # This puts the entire model, including encoder, in training mode.

    # <<< MODIFIED: Conditionally set encoder back to eval() mode only if not fine-tuning.
    # This is the key change for the training loop.
    if not config_params.get('FINE_TUNE_ENCODER', False):
        model.encoder.eval()

    total_policy_loss, total_value_loss = 0, 0
    np.random.shuffle(game_memory)

    num_batches = (len(game_memory) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Training on game batches", leave=False):
        batch_memory_slice = game_memory[i * batch_size : (i + 1) * batch_size]
        if not batch_memory_slice:
            continue

        gnn_batch, cnn_batch, target_policies, target_values = prepare_sequence_batch(batch_memory_slice, device)

        optimizer.zero_grad()
        policy_logits, value_preds = model(gnn_batch, cnn_batch)

        policy_loss = -(torch.nn.functional.log_softmax(policy_logits, dim=1) * target_policies).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(-1), target_values.squeeze(-1))
        total_loss = policy_loss + value_loss

        total_loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / num_batches, total_value_loss / num_batches

def save_checkpoint(model: TemporalPolicyValueModel, optimizer: optim.Optimizer, game_number: int, directory: Path):
    checkpoint_path = directory / f"temporal_checkpoint_game_{game_number}.pth.tar"
    torch.save({
        'game_number': game_number,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="Run the Phase D training loop with a Temporal Transformer.")
    parser.add_argument('--encoder_checkpoint', type=str, required=True, help="Path to a PRE-TRAINED ENCODER model from Phase C.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a TEMPORAL model checkpoint to continue training.")
    args = parser.parse_args()

    paths = get_paths()
    device = get_device()
    print(f"Using device: {device}")

    encoder = EncoderPolicyValueModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
        embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
    ).to(device)

    encoder_checkpoint_path = Path(args.encoder_checkpoint)
    if not encoder_checkpoint_path.exists():
        print(f"[FATAL] Encoder checkpoint not found at {encoder_checkpoint_path}.")
        sys.exit(1)

    print(f"Loading pre-trained encoder from: {encoder_checkpoint_path}")
    encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=device)
    encoder_state_dict = encoder_checkpoint['model_state_dict']

    if 'policy_head.weight' in encoder_state_dict:
        encoder_state_dict.pop('policy_head.weight')
    if 'policy_head.bias' in encoder_state_dict:
        encoder_state_dict.pop('policy_head.bias')
    
    encoder.load_state_dict(encoder_state_dict, strict=False)
    print("INFO: Successfully loaded compatible weights from encoder checkpoint.")

    # <<< MODIFIED: Pass the fine_tune_encoder flag from config to the model.
    model = TemporalPolicyValueModel(
        encoder_model=encoder,
        policy_size=get_action_space_size(),
        d_model=config_params['EMBED_DIM'],
        fine_tune_encoder=config_params.get('FINE_TUNE_ENCODER', False)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config_params['LEARNING_RATE'], weight_decay=config_params['WEIGHT_DECAY'])
    
    start_game = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading temporal model checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_game = checkpoint.get('game_number', 0)
            print(f"Resuming from game {start_game + 1}")
        else:
            print(f"[WARNING] Temporal checkpoint not found at {checkpoint_path}. Starting fresh.")
    else:
        print("No temporal checkpoint provided. Starting fresh.")

    mcts_player = MCTS(network=model, device=device, c_puct=config_params['CPUCT'], batch_size=config_params['BATCH_SIZE'])

    try:
        stockfish_params = {"Skill Level": config_params.get('STOCKFISH_ELO', 1400)}
        stockfish_opponent = Stockfish(path=config_params['STOCKFISH_PATH'], parameters=stockfish_params)
        stockfish_depth = config_params.get('STOCKFISH_DEPTH', 5)
        print(f"Initialized Stockfish with Elo: {stockfish_params['Skill Level']} and Depth: {stockfish_depth}")
    except Exception as e:
        print(f"[FATAL] Could not initialize the Stockfish engine: {e}"); sys.exit(1)

    # ... (The rest of the main loop is unchanged) ...
    print("\n" + "#"*60 + "\n--- Phase D: Temporal Transformer Training Begins ---\n" + "#"*60 + "\n")
    if config_params.get('FINE_TUNE_ENCODER', False):
        print("--- RUNNING IN FINE-TUNING MODE ---")


    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        print(f"\n--- Starting Game {game_num}/{config_params['TOTAL_GAMES']} ---")
        board = chess.Board()
        game_memory: List[GameStep] = []
        agent_is_white = (game_num % 2 == 1)
        print(f"Agent is playing as {'WHITE' if agent_is_white else 'BLACK'}")

        initial_gnn, initial_cnn, _ = convert_to_gnn_input(chess.Board(), device)
        initial_state_tuple = (initial_gnn, initial_cnn)
        state_sequence_queue = deque([initial_state_tuple] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)

        game = chess.pgn.Game()
        game.headers["Event"] = "Phase D Temporal Sparring (Fine-Tuning)" if config_params.get('FINE_TUNE_ENCODER', False) else "Phase D Temporal Sparring (Frozen)"
        game.headers["Site"] = "Colab Environment"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(game_num)
        game.headers["White"] = "Agent" if agent_is_white else "Stockfish"
        game.headers["Black"] = "Stockfish" if agent_is_white else "Agent"
        current_node = game

        while not board.is_game_over(claim_draw=True):
            is_agent_turn = (board.turn == chess.WHITE and agent_is_white) or (board.turn == chess.BLACK and not agent_is_white)

            if is_agent_turn:
                policy_dict, mcts_value = mcts_player.run_search(board, config_params['MCTS_SIMULATIONS'], state_sequence=list(state_sequence_queue))
                if not policy_dict: break
                move = mcts_player.select_move(policy_dict, temperature=1.0)
                policy_tensor = format_policy_for_training(policy_dict, board)
                value_target_tensor = torch.tensor([mcts_value], dtype=torch.float32)
                game_memory.append({ "state_sequence": list(state_sequence_queue), "policy": policy_tensor, "value_target": value_target_tensor})
                san_move = board.san(move)
                print(f"Agent plays: {san_move} (MCTS Value: {mcts_value:.4f})")
                board.push(move)
                new_gnn, new_cnn, _ = convert_to_gnn_input(board, device)
                state_sequence_queue.append((new_gnn, new_cnn))
            else:
                stockfish_opponent.set_fen_position(board.fen())
                best_move_uci = stockfish_opponent.get_best_move()
                if not best_move_uci: break
                move = chess.Move.from_uci(best_move_uci)
                san_move = board.san(move)
                print(f"Stockfish plays: {san_move}")
                board.push(move)
                new_gnn, new_cnn, _ = convert_to_gnn_input(board, device)
                state_sequence_queue.append((new_gnn, new_cnn))
            
            current_node = current_node.add_main_variation(move)

        result_str = board.result(claim_draw=True)
        game.headers["Result"] = result_str
        print(f"--- Game Over. Result: {result_str} ---")

        pgn_filename = paths.pgn_games_dir / f"temporal_sparring_game_{game_num}.pgn"
        with open(pgn_filename, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)
        print(f"PGN for game saved to: {pgn_filename}")

        if game_memory:
            policy_loss, value_loss = train_on_game(model, optimizer, game_memory, config_params['BATCH_SIZE'], device)
            print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")

        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            save_checkpoint(model, optimizer, game_num, paths.checkpoints_dir)

    print("\n--- Training Run Finished ---")


if __name__ == "__main__":
    main()