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
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size
from hardware_setup import get_device

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StyleDataset(Dataset):
    """
    PyTorch Dataset for loading chess positions from a .jsonl file,
    converting them to tensors, and handling different style labels.
    """
    def __init__(self, jsonl_path: Path):
        self.positions = []
        logging.info(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Reading dataset file"):
                # Store the raw JSON line to parse in __getitem__
                self.positions.append(json.loads(line))
        logging.info(f"Successfully loaded {len(self.positions)} positions.")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        """
        Retrieves a single data point, converts it to tensor format, and assigns a label.
        """
        data_point = self.positions[idx]
        fen = data_point['fen']
        label_str = data_point['label']
        
        board = chess.Board(fen)
        gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, torch.device('cpu'))
        
        # --- MODIFIED: Convert string label to numeric label ---
        # "petrosian_win" (safe style) -> 1.0
        # "tal_win" (risky style) -> 0.0
        label = 1.0 if "petrosian" in label_str else 0.0
        
        return gnn_data, cnn_tensor, torch.tensor(label, dtype=torch.float32)

class StyleClassifierModel(nn.Module):
    """
    A classifier that uses the frozen GNN+CNN base from our main agent
    and adds a new head to classify a position's style.
    """
    def __init__(self, base_model: PolicyValueModel):
        super().__init__()
        self.base_model = base_model
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.classifier_head = nn.Sequential(
            nn.Linear(base_model.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, gnn_batch: Batch, cnn_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = cnn_tensor.size(0)
        
        with torch.no_grad():
            gnn_out = self.base_model.gnn(gnn_batch)
            cnn_out = self.base_model.cnn(cnn_tensor)
            gnn_out_reshaped = gnn_out.view(batch_size, 64, self.base_model.embed_dim).mean(dim=1)
            cnn_out_pooled = cnn_out.view(batch_size, self.base_model.embed_dim, -1).mean(dim=2)
            fused = torch.cat([gnn_out_reshaped, cnn_out_pooled], dim=-1)
            final_embedding = self.base_model.embedding_projection(fused)
        
        style_logit = self.classifier_head(final_embedding)
        return style_logit

def main():
    parser = argparse.ArgumentParser(description="Train a style classifier on a dataset of FENs.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--base-checkpoint', type=str, required=True, help="Path to the agent checkpoint to use as the base feature extractor.")
    args = parser.parse_args()

    device = get_device()
    paths = get_paths()
    # --- MODIFIED: Point to the new combined dataset ---
    dataset_path = paths.training_data_dir / "combined_style_dataset.jsonl"

    if not dataset_path.exists():
        logging.error(f"FATAL: Dataset not found at {dataset_path}"); sys.exit(1)

    full_dataset = StyleDataset(dataset_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    def collate_fn(batch):
        gnn_data, cnn_tensors, labels = zip(*batch)
        gnn_batch = Batch.from_data_list(list(gnn_data))
        cnn_batch = torch.stack(cnn_tensors, 0)
        labels_batch = torch.stack(labels, 0)
        return gnn_batch, cnn_batch, labels_batch

    train_loader = DataLoader(train_dataset, batch_size=config_params['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config_params['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn)

    logging.info(f"Loading base model from {args.base_checkpoint} to use as a feature extractor.")
    base_model = PolicyValueModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14, 
        embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'],
        gnn_metadata=(['square', 'piece'], [('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')])
    ).to(device)
    
    checkpoint = torch.load(args.base_checkpoint, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])

    model = StyleClassifierModel(base_model).to(device)
    optimizer = optim.Adam(model.classifier_head.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    best_val_accuracy = 0.0
    logging.info("--- Starting Classifier Training on Combined Dataset ---")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for gnn_batch, cnn_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]"):
            gnn_batch, cnn_batch, labels = gnn_batch.to(device), cnn_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(gnn_batch, cnn_batch).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for gnn_batch, cnn_batch, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]"):
                gnn_batch, cnn_batch, labels = gnn_batch.to(device), cnn_batch.to(device), labels.to(device)
                logits = model(gnn_batch, cnn_batch).squeeze(1)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                preds = torch.sigmoid(logits) > 0.5
                correct_predictions += (preds == labels.byte()).sum().item()
                total_samples += labels.size(0)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_samples

        logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = paths.checkpoints_dir / "best_petrosian_vs_tal_classifier.pth"
            torch.save(model.classifier_head.state_dict(), save_path)
            logging.info(f"New best model saved to {save_path} with accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()