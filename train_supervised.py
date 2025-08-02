import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
from torch_geometric.data import Batch
import chess

# Add project root to path to allow importing from our modules
import sys
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config import get_paths, config_params
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from hardware_setup import get_device

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SupervisedChessDataset(Dataset):
    """
    PyTorch Dataset for the supervised learning task of predicting a grandmaster's move.
    """
    def __init__(self, jsonl_path: Path):
        self.data_points = []
        logging.info(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Reading dataset file"):
                self.data_points.append(json.loads(line))
        logging.info(f"Successfully loaded {len(self.data_points)} positions.")

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        """
        Retrieves a single data point (board state and played move) and converts it to tensors.
        """
        data_point = self.data_points[idx]
        fen = data_point['fen']
        uci_move = data_point['played_move']
        
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)

        gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, torch.device('cpu'))
        target_policy_index = move_to_index(move, board)
        
        return gnn_data, cnn_tensor, torch.tensor(target_policy_index, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Run a supervised training job for the PolicyValueModel.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    args = parser.parse_args()

    device = get_device()
    paths = get_paths()
    dataset_path = paths.training_data_dir / "bedrock_petrosian_dataset.jsonl"

    if not dataset_path.exists():
        logging.error(f"FATAL: Dataset not found at {dataset_path}"); sys.exit(1)

    # --- 1. Data Loading ---
    full_dataset = SupervisedChessDataset(dataset_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    def collate_fn(batch):
        gnn_data, cnn_tensors, labels = zip(*batch)
        gnn_batch = Batch.from_data_list(list(gnn_data))
        cnn_batch = torch.stack(cnn_tensors, 0)
        labels_batch = torch.stack(labels, 0)
        return gnn_batch, cnn_batch, labels_batch

    train_loader = DataLoader(train_dataset, batch_size=config_params['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config_params['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn, num_workers=2)

    # --- 2. Model Initialization ---
    logging.info("Initializing a new PolicyValueModel from scratch.")
    model = PolicyValueModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14, 
        embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'],
        gnn_metadata=(['square', 'piece'], [('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')])
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config_params['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss()

    # --- 3. Training & Validation Loop ---
    best_val_accuracy = 0.0
    logging.info("--- Starting Supervised Training (Bedrock Experiment) ---")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for gnn_batch, cnn_batch, target_indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            gnn_batch, cnn_batch, target_indices = gnn_batch.to(device), cnn_batch.to(device), target_indices.to(device)
            
            optimizer.zero_grad()
            policy_logits, _ = model(gnn_batch, cnn_batch) # We ignore the value head output
            loss = criterion(policy_logits, target_indices)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for gnn_batch, cnn_batch, target_indices in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                gnn_batch, cnn_batch, target_indices = gnn_batch.to(device), cnn_batch.to(device), target_indices.to(device)
                policy_logits, _ = model(gnn_batch, cnn_batch)
                loss = criterion(policy_logits, target_indices)
                total_val_loss += loss.item()

                _, predicted_indices = torch.max(policy_logits, 1)
                correct_predictions += (predicted_indices == target_indices).sum().item()
                total_samples += target_indices.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples

        logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        # --- 4. Model Saving ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = paths.checkpoints_dir / "bedrock_supervised_model.pth"
            torch.save(model.state_dict(), save_path)
            logging.info(f"New best model saved to {save_path} with accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()