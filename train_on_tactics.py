#
# File: train_on_tactics.py (Corrected for Phase AO)
#
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.optim as optim
import chess
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

# --- Setup Python Path ---
# This allows us to import from the gnn_agent package
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    # In this script, the project root is the current directory.
    # For other scripts, it might be parent.parent.
    sys.path.insert(0, str(project_root))

# --- Project-specific Imports ---
from config import get_paths
from gnn_agent.neural_network.gnn_models import ChessNetwork
from gnn_agent.gamestate_converters import gnn_data_converter
from gnn_agent.gamestate_converters.action_space_converter import move_to_index

# --- Dataset Class ---

class TacticalPuzzleDataset(Dataset):
    """A PyTorch Dataset for loading tactical puzzles from a .jsonl file."""

    def __init__(self, jsonl_file_path: str):
        self.puzzles = []
        if not os.path.exists(jsonl_file_path):
            raise FileNotFoundError(
                f"Puzzle file not found at {jsonl_file_path}. "
                "Please ensure the file exists in the root directory."
            )
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                self.puzzles.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.puzzles[idx]

# --- Core Functions ---

def create_new_network(device: torch.device) -> ChessNetwork:
    """
    Creates a new, randomly initialized ChessNetwork with the latest architecture.
    """
    print("✅ Creating a new, randomly initialized network for tactical training...")
    network = ChessNetwork(
        embed_dim=256,
        gnn_hidden_dim=128,
        num_heads=4
    ).to(device)
    return network

def collate_puzzles(batch: List[Dict[str, Any]]) -> Optional[Tuple[Batch, torch.Tensor]]:
    """
    Custom collate function to process a batch of puzzles into a single
    PyTorch Geometric Batch object. This version also constructs the
    necessary 'piece_batch' tensor for correct pooling.
    """
    gnn_data_list = []
    policy_targets = []
    
    for puzzle in batch:
        fen = puzzle['fen']
        best_move_uci = puzzle['best_move']
        
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(best_move_uci)
            
            if move in board.legal_moves:
                gnn_data = gnn_data_converter.convert_to_gnn_input(board, device='cpu')
                move_idx = move_to_index(move, board)
                
                gnn_data_list.append(gnn_data)
                policy_targets.append(move_idx)
            else:
                print(f"Warning: Skipping illegal move {best_move_uci} for FEN {fen}.")

        except (ValueError, AssertionError) as e:
            print(f"Warning: Skipping puzzle due to error. FEN: {fen}, Move: {best_move_uci}. Reason: {e}")

    if not gnn_data_list:
        return None

    # Use PyG's Batch class to correctly batch graph data
    batched_gnn_data = Batch.from_data_list(gnn_data_list)
    
    # --- BUG FIX: Create and attach the batch vector for the pieces ---
    # The default 'batch' attribute corresponds to the square nodes.
    # We need to create one for the piece nodes to use for pooling.
    piece_batch_list = []
    for i, data in enumerate(gnn_data_list):
        num_pieces = data.piece_features.size(0)
        piece_batch_list.append(torch.full((num_pieces,), i, dtype=torch.long))
    
    if piece_batch_list:
        batched_gnn_data.piece_batch = torch.cat(piece_batch_list)
    else:
        batched_gnn_data.piece_batch = torch.empty((0,), dtype=torch.long)
    # --- END BUG FIX ---

    policy_targets_tensor = torch.tensor(policy_targets, dtype=torch.long)

    return batched_gnn_data, policy_targets_tensor


def main():
    """Main function to run the tactical training session."""
    parser = argparse.ArgumentParser(description="Run supervised training on tactical puzzles from scratch.")
    parser.add_argument("--puzzles_path", type=str, required=True, help="Path to the tactical_puzzles.jsonl file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the new model checkpoint.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()

    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    network = create_new_network(device)
    network.train()

    try:
        dataset = TacticalPuzzleDataset(args.puzzles_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    if len(dataset) == 0:
        print("Puzzle file is empty. Nothing to train on. Exiting.")
        return
        
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_puzzles)
    print(f"Loaded {len(dataset)} puzzles. Starting training for {args.epochs} epochs.")

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    policy_loss_fn = torch.nn.NLLLoss()

    for epoch in range(args.epochs):
        total_policy_loss = 0
        puzzles_processed = 0

        for batch_data in data_loader:
            if batch_data is None:
                continue
            
            gnn_batch, policy_targets_batch = batch_data

            optimizer.zero_grad()
            
            gnn_batch = gnn_batch.to(device)
            policy_targets_batch = policy_targets_batch.to(device)
            
            policy_logits, _ = network(gnn_batch)
            
            loss = policy_loss_fn(policy_logits, policy_targets_batch)
            
            loss.backward()
            optimizer.step()
            
            total_policy_loss += loss.item() * gnn_batch.num_graphs
            puzzles_processed += gnn_batch.num_graphs

        avg_epoch_loss = total_policy_loss / puzzles_processed if puzzles_processed > 0 else 0
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Policy Loss: {avg_epoch_loss:.6f}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs,
        'final_loss': avg_epoch_loss
    }, output_path)
    
    print(f"\n✅ Training complete. Saved new model to:\n{output_path}")

if __name__ == '__main__':
    main()
