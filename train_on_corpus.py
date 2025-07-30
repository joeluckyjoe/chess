# FILENAME: train_on_corpus.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
import chess
import json
import logging
from tqdm import tqdm
from pathlib import Path
# MODIFIED: Import argparse for command-line arguments
import argparse

# --- Project-specific Imports ---
from config import get_paths, config_params
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import move_to_index, get_action_space_size

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Ensure logs go to console
)

# --- GNN Metadata ---
GNN_METADATA = (
    ['square', 'piece'],
    [
        ('square', 'adjacent_to', 'square'),
        ('piece', 'occupies', 'square'),
        ('piece', 'attacks', 'piece'),
        ('piece', 'defends', 'piece')
    ]
)

# --- PyTorch Dataset for Supervised Learning ---
class SupervisedChessDataset(Dataset):
    """Loads (FEN, move, outcome) data from the generated JSONL file."""
    def __init__(self, jsonl_path: Path):
        self.data = []
        logging.info(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Reading dataset"):
                self.data.append(json.loads(line))
        logging.info(f"Loaded {len(self.data)} positions.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        fen = item['fen']
        uci_move = item['played_move']
        outcome = item['outcome']

        board = chess.Board(fen)
        try:
            move = chess.Move.from_uci(uci_move)
        except chess.InvalidMoveError:
            logging.warning(f"Invalid UCI move '{uci_move}' for FEN '{fen}'. Skipping.")
            return None

        gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device='cpu')

        policy_target = move_to_index(move, board)
        value_target = torch.tensor([outcome], dtype=torch.float32)

        if policy_target is None:
            # This can happen if the move is legal but not in our canonical action space (e.g., rare promotions)
            logging.warning(f"Could not find index for move {uci_move} on board {fen}. Skipping.")
            return None

        return gnn_data, cnn_tensor, torch.tensor(policy_target, dtype=torch.long), value_target

def collate_fn(batch):
    """Custom collate function to handle HeteroData batching."""
    batch = list(filter(None, batch))
    if not batch:
        return None, None, None, None
    gnn_list, cnn_list, policy_list, value_list = zip(*batch)
    gnn_batch = Batch.from_data_list(gnn_list)
    cnn_batch = torch.stack(cnn_list)
    policy_batch = torch.stack(policy_list)
    value_batch = torch.stack(value_list)
    return gnn_batch, cnn_batch, policy_batch, value_batch

# --- Main Training and Evaluation Loop ---
def train_on_corpus(input_file: str, output_file: str, epochs: int):
    """Main function to orchestrate the supervised training process."""
    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = ValueNextStateModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'],
        cnn_in_channels=14,
        embed_dim=config_params['EMBED_DIM'],
        policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'],
        gnn_metadata=GNN_METADATA
    ).to(device)

    logging.info("Initializing lazy layers by performing a dummy forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            dummy_board = chess.Board()
            gnn_data, cnn_tensor, _ = convert_to_gnn_input(dummy_board, device='cpu')
            dummy_gnn_batch = Batch.from_data_list([gnn_data]).to(device)
            dummy_cnn_tensor = torch.stack([cnn_tensor]).to(device)
            _ = model(dummy_gnn_batch, dummy_cnn_tensor)
        logging.info("Model layers initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize lazy layers. Error: {e}", exc_info=True)
        exit()

    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} trainable parameters.")

    # MODIFIED: Use the dataset path from the command-line argument
    dataset_path = paths.drive_project_root / input_file
    if not dataset_path.exists():
        logging.error(f"Dataset not found at {dataset_path}. Please run create_corpus.py with the correct PGN to generate the JSONL file.")
        return

    dataset = SupervisedChessDataset(dataset_path)

    if len(dataset) == 0:
        logging.error("Dataset is empty. Aborting training.")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config_params['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config_params['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=config_params['LEARNING_RATE'], weight_decay=config_params['WEIGHT_DECAY'])
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    best_val_loss = float('inf')

    logging.info(f"Starting supervised training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_train_loss, total_policy_loss, total_value_loss, total_next_state_loss = 0, 0, 0, 0
        for gnn_batch, cnn_batch, policy_targets, value_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            if gnn_batch is None: continue
            gnn_batch, cnn_batch, policy_targets, value_targets = gnn_batch.to(device), cnn_batch.to(device), policy_targets.to(device), value_targets.to(device)
            
            optimizer.zero_grad()
            
            # MODIFIED: Unpack all three head outputs
            policy_logits, value, next_state_value = model(gnn_batch, cnn_batch)
            
            loss_policy = policy_loss_fn(policy_logits, policy_targets)
            loss_value = value_loss_fn(value.squeeze(-1), value_targets.squeeze(-1))
            # MODIFIED: Calculate loss for the NextStateValue head
            loss_next_state = value_loss_fn(next_state_value.squeeze(-1), value_targets.squeeze(-1))
            
            # MODIFIED: Include all three losses in the total loss
            total_loss = loss_policy + config_params['VALUE_LOSS_WEIGHT'] * (loss_value + loss_next_state)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Grad clipping
            optimizer.step()
            
            total_train_loss += total_loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            total_next_state_loss += loss_next_state.item()

        avg_policy_loss = total_policy_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
        avg_next_state_loss = total_next_state_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} Train | Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Next-State Loss: {avg_next_state_loss:.4f}")

        model.eval()
        total_val_loss = 0
        correct_policy_preds, total_policy_preds = 0, 0
        with torch.no_grad():
            for gnn_batch, cnn_batch, policy_targets, value_targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if gnn_batch is None: continue
                gnn_batch, cnn_batch, policy_targets, value_targets = gnn_batch.to(device), cnn_batch.to(device), policy_targets.to(device), value_targets.to(device)
                
                # MODIFIED: Unpack all three head outputs
                policy_logits, value, next_state_value = model(gnn_batch, cnn_batch)
                
                loss_policy = policy_loss_fn(policy_logits, policy_targets)
                loss_value = value_loss_fn(value.squeeze(-1), value_targets.squeeze(-1))
                # MODIFIED: Calculate loss for the NextStateValue head
                loss_next_state = value_loss_fn(next_state_value.squeeze(-1), value_targets.squeeze(-1))
                
                # MODIFIED: Include all three losses in the total validation loss
                total_loss = loss_policy + config_params['VALUE_LOSS_WEIGHT'] * (loss_value + loss_next_state)
                total_val_loss += total_loss.item()

                _, predicted_moves = torch.max(policy_logits, 1)
                correct_policy_preds += (predicted_moves == policy_targets).sum().item()
                total_policy_preds += policy_targets.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        policy_accuracy = (correct_policy_preds / total_policy_preds) * 100
        logging.info(f"Epoch {epoch+1} Val   | Avg Total Loss: {avg_val_loss:.4f}, Policy Accuracy: {policy_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # MODIFIED: Use the output file name from the command-line argument
            checkpoint_path = paths.checkpoints_dir / output_file
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"New best model saved to {checkpoint_path} with validation loss {best_val_loss:.4f}")

    logging.info("Supervised training finished.")

# MODIFIED: Add argument parser to define inputs and outputs
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a chess model on a corpus of games.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Name of the input .jsonl file (e.g., 'RuyLopezBerlin_supervised_dataset.jsonl') located in the project's root on Drive."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Name of the output model checkpoint file (e.g., 'phase_by_ruy_lopez_specialist.pth') to be saved in the checkpoints directory."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to train for."
    )
    args = parser.parse_args()
    
    train_on_corpus(args.input_file, args.output_file, args.epochs)