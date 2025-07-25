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

# --- Project-specific Imports ---
# These assume your project structure allows these imports from the root directory.
# You may need to adjust them based on your PYTHONPATH setup.
from config import get_paths, config_params
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Ensure logs go to console
)

# --- GNN Metadata and Move Handler ---
# This metadata is required to initialize the UnifiedGNN within your model.
GNN_METADATA = (
    ['square', 'piece'],
    [
        ('square', 'adjacent_to', 'square'),
        ('piece', 'occupies', 'square'),
        ('piece', 'attacks', 'piece'),
        ('piece', 'defends', 'piece')
    ]
)

class MoveHandler:
    """
    A self-contained class to handle move-to-index and index-to-move conversions.
    This is a standard implementation for a 4672-size policy head.
    """
    def __init__(self):
        self.move_map = self._create_move_map()
        self.inv_move_map = {v: k for k, v in self.move_map.items()}
        self.policy_size = len(self.move_map)

    def _create_move_map(self):
        moves = {}
        idx = 0
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq == to_sq:
                    continue
                # Queen promotions
                for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves[chess.Move(from_sq, to_sq, promotion=prom)] = idx
                    idx += 1
                # Non-promotion moves
                moves[chess.Move(from_sq, to_sq)] = idx
                idx += 1
        return moves

    def move_to_index(self, move: chess.Move) -> int:
        # A simple move might not have promotion info, find the base move key
        base_move = chess.Move(move.from_square, move.to_square, promotion=None)
        if move.promotion is not None:
             return self.move_map.get(move)
        return self.move_map.get(base_move)


# --- PyTorch Dataset for Supervised Learning ---

class SupervisedChessDataset(Dataset):
    """Loads (FEN, move, outcome) data from the generated JSONL file."""
    def __init__(self, jsonl_path: Path, move_handler: MoveHandler):
        self.move_handler = move_handler
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
        move = chess.Move.from_uci(uci_move)

        # Convert data for the model
        # Note: We run this on 'cpu' and let the DataLoader handle GPU transfer
        gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device='cpu')
        
        policy_target = self.move_handler.move_to_index(move)
        value_target = torch.tensor([outcome], dtype=torch.float32)

        if policy_target is None:
            logging.warning(f"Could not find index for move {uci_move} on board {fen}. Skipping.")
            return None

        return gnn_data, cnn_tensor, torch.tensor(policy_target, dtype=torch.long), value_target

def collate_fn(batch):
    """Custom collate function to handle HeteroData batching."""
    batch = list(filter(None, batch)) # Remove None items from failed move lookups
    if not batch:
        return None, None, None, None

    gnn_list, cnn_list, policy_list, value_list = zip(*batch)
    
    gnn_batch = Batch.from_data_list(gnn_list)
    cnn_batch = torch.stack(cnn_list)
    policy_batch = torch.stack(policy_list)
    value_batch = torch.stack(value_list)
    
    return gnn_batch, cnn_batch, policy_batch, value_batch

# --- Main Training and Evaluation Loop ---

def train_on_corpus():
    """
    Main function to orchestrate the supervised training process.
    """
    # --- 1. Setup ---
    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    move_handler = MoveHandler()

    # --- 2. Model Initialization ---
    model = ValueNextStateModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'],
        cnn_in_channels=14, # From gnn_data_converter.py
        embed_dim=config_params['EMBED_DIM'],
        policy_size=move_handler.policy_size,
        gnn_num_heads=config_params['GNN_NUM_HEADS'],
        gnn_metadata=GNN_METADATA
    ).to(device)

    # --- ADDED: Initialize lazy layers with a dummy forward pass ---
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
        # Exit if initialization fails, as the rest of the script will also fail.
        exit()
    # --- END OF ADDED BLOCK ---

    # MOVED: This line now comes AFTER the initialization.
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} trainable parameters.")

    # --- 3. Data Loading ---
    dataset_path = paths.drive_project_root / 'kasparov_supervised_dataset.jsonl'
    if not dataset_path.exists():
        logging.error(f"Dataset not found at {dataset_path}. Please run create_supervised_dataset.py first.")
        return

    dataset = SupervisedChessDataset(dataset_path, move_handler)

    # Split dataset: 90% training, 10% validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_params['BATCH_SIZE'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2, # Adjust based on your system
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_params['BATCH_SIZE'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # --- 4. Training Configuration ---
    optimizer = optim.AdamW(model.parameters(), lr=config_params['LEARNING_RATE'], weight_decay=config_params['WEIGHT_DECAY'])
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    epochs = 10 # Define number of training epochs

    logging.info("Starting supervised training...")

    # --- 5. Training Loop ---
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_train_loss, total_policy_loss, total_value_loss = 0, 0, 0
        for gnn_batch, cnn_batch, policy_targets, value_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            if gnn_batch is None: continue

            gnn_batch = gnn_batch.to(device)
            cnn_batch = cnn_batch.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            optimizer.zero_grad()
            
            # Forward pass - ignore the third output (next_state_value)
            policy_logits, value, _ = model(gnn_batch, cnn_batch)

            loss_policy = policy_loss_fn(policy_logits, policy_targets)
            loss_value = value_loss_fn(value.squeeze(-1), value_targets.squeeze(-1))
            
            total_loss = loss_policy + loss_value * config_params['VALUE_LOSS_WEIGHT']
            
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_policy_loss = total_policy_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} Train | Avg Total Loss: {avg_train_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

        # Validation Phase
        model.eval()
        total_val_loss, correct_policy_preds, total_policy_preds = 0, 0, 0
        with torch.no_grad():
            for gnn_batch, cnn_batch, policy_targets, value_targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if gnn_batch is None: continue

                gnn_batch = gnn_batch.to(device)
                cnn_batch = cnn_batch.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)

                policy_logits, value, _ = model(gnn_batch, cnn_batch)
                
                loss_policy = policy_loss_fn(policy_logits, policy_targets)
                loss_value = value_loss_fn(value.squeeze(-1), value_targets.squeeze(-1))
                total_loss = loss_policy + loss_value * config_params['VALUE_LOSS_WEIGHT']
                
                total_val_loss += total_loss.item()
                
                # Calculate policy accuracy
                _, predicted_moves = torch.max(policy_logits, 1)
                correct_policy_preds += (predicted_moves == policy_targets).sum().item()
                total_policy_preds += policy_targets.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        policy_accuracy = (correct_policy_preds / total_policy_preds) * 100
        logging.info(f"Epoch {epoch+1} Val   | Avg Total Loss: {avg_val_loss:.4f}, Policy Accuracy: {policy_accuracy:.2f}%")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = paths.checkpoints_dir / "pretrained_kasparov_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"New best model saved to {checkpoint_path} with validation loss {best_val_loss:.4f}")

    logging.info("Supervised training finished.")


if __name__ == '__main__':
    train_on_corpus()