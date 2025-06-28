#
# File: train_on_tactics.py
#
"""
A standalone script for targeted training of the ChessNetwork on a dataset of
tactical puzzles.

This script implements the supervised learning portion of Phase O. It loads a
file of puzzles (FEN and best move), and trains the policy head of the network
to predict the correct move. This is intended to fix the agent's "tactical
blindness" by burning in recognition of forced-mate patterns.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.optim as optim
import chess
from torch.utils.data import Dataset, DataLoader

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Project-specific Imports ---
from config import get_paths
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters import gnn_data_converter

# --- Constants ---
# The total number of possible moves in the action space, derived from the map.
# This must match the output dimension of the PolicyHead.
TOTAL_POSSIBLE_MOVES = 4672 

# --- Dataset Class ---

class TacticalPuzzleDataset(Dataset):
    """A PyTorch Dataset for loading tactical puzzles from a .jsonl file."""

    def __init__(self, jsonl_file_path):
        self.puzzles = []
        if not os.path.exists(jsonl_file_path):
            raise FileNotFoundError(
                f"Puzzle file not found at {jsonl_file_path}. "
                "Please generate it first using export_game_analysis.py on games with checkmates."
            )
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                self.puzzles.append(json.loads(line))

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return self.puzzles[idx]

# --- Core Functions ---

def find_latest_checkpoint(checkpoint_dir):
    """Finds the most recently modified checkpoint file in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth.tar'))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def load_model_from_checkpoint(model_path, device):
    """Loads a ChessNetwork model and optimizer state from a .pth checkpoint file."""
    square_gnn = SquareGNN(in_features=12, hidden_features=256, out_features=128, heads=4)
    piece_gnn = PieceGNN(in_channels=12, hidden_channels=256, out_channels=128)
    attention_module = CrossAttentionModule(sq_embed_dim=128, pc_embed_dim=128, num_heads=4, dropout_rate=0.1)
    policy_head = PolicyHead(embedding_dim=128, num_possible_moves=TOTAL_POSSIBLE_MOVES)
    value_head = ValueHead(embedding_dim=128)
    network = ChessNetwork(square_gnn, piece_gnn, attention_module, policy_head, value_head).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    # For this training, we only care about the model weights, not the optimizer state from the RL loop.
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    network.load_state_dict(state_dict)
    print(f"Successfully loaded model weights from {os.path.basename(str(model_path))}")
    return network

def collate_puzzles(batch):
    """Custom collate function to process a batch of puzzles."""
    gnn_inputs = []
    policy_targets = []
    
    for puzzle in batch:
        fen = puzzle['fen']
        best_move_uci = puzzle['best_move']
        
        board = chess.Board(fen)
        move = chess.Move.from_uci(best_move_uci)

        # Use the converters to get model input and policy target
        gnn_input = gnn_data_converter.convert_to_gnn_input(board, device='cpu')
        
        # This relies on the move map from the converter file
        move_index = gnn_data_converter.MOVE_TO_INDEX_MAP.get(move)

        if move_index is not None:
            gnn_inputs.append(gnn_input)
            policy_targets.append(move_index)

    return gnn_inputs, torch.tensor(policy_targets, dtype=torch.long)


def main():
    """Main function to run the tactical training session."""
    parser = argparse.ArgumentParser(description="Run supervised training on tactical puzzles.")
    parser.add_argument("--puzzles_path", type=str, required=True, help="Path to the tactical_puzzles.jsonl file.")
    parser.add_argument("--model_path", type=str, default=None, help="(Optional) Path to a specific model checkpoint. If not provided, the latest checkpoint will be used.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    args = parser.parse_args()

    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model_path = args.model_path
    if not model_path:
        print("Searching for the latest checkpoint...")
        model_path = find_latest_checkpoint(paths.checkpoints_dir)
        if not model_path:
            print(f"Error: No checkpoints found in '{paths.checkpoints_dir}'. Exiting.")
            return
    
    network = load_model_from_checkpoint(model_path, device)
    network.train() # Set model to training mode

    # --- Load Data ---
    try:
        dataset = TacticalPuzzleDataset(args.puzzles_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    if len(dataset) == 0:
        print("Puzzle file is empty. Nothing to train on. Exiting.")
        return
        
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_puzzles)
    print(f"Loaded {len(dataset)} puzzles. Starting training for {args.epochs} epochs.")

    # --- Training Setup ---
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    # CrossEntropyLoss is perfect for single-correct-answer classification, which is what this is.
    policy_loss_fn = torch.nn.CrossEntropyLoss()

    # --- Training Loop ---
    for epoch in range(args.epochs):
        total_policy_loss = 0
        puzzles_processed = 0

        for gnn_inputs_batch, policy_targets_batch in data_loader:
            optimizer.zero_grad()
            
            # Move targets to the correct device
            policy_targets_batch = policy_targets_batch.to(device)
            
            # Process each item in the batch individually
            # (A more optimized version could truly batch the graph data, but this is safer and clearer)
            batch_policy_loss = 0
            for i, gnn_input in enumerate(gnn_inputs_batch):
                # Move individual graph tensors to the correct device
                gnn_input.square_graph.x = gnn_input.square_graph.x.to(device)
                gnn_input.square_graph.edge_index = gnn_input.square_graph.edge_index.to(device)
                gnn_input.piece_graph.x = gnn_input.piece_graph.x.to(device)
                gnn_input.piece_graph.edge_index = gnn_input.piece_graph.edge_index.to(device)
                gnn_input.piece_to_square_map = gnn_input.piece_to_square_map.to(device)

                policy_logits, _ = network(*gnn_input) # We only need policy logits
                
                # Reshape for loss calculation: (1, NumClasses) and (1)
                loss = policy_loss_fn(policy_logits.unsqueeze(0), policy_targets_batch[i].unsqueeze(0))
                batch_policy_loss += loss

            if gnn_inputs_batch:
                avg_loss = batch_policy_loss / len(gnn_inputs_batch)
                avg_loss.backward()
                optimizer.step()
                
                total_policy_loss += avg_loss.item() * len(gnn_inputs_batch)
                puzzles_processed += len(gnn_inputs_batch)

        avg_epoch_loss = total_policy_loss / puzzles_processed if puzzles_processed > 0 else 0
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Policy Loss: {avg_epoch_loss:.6f}")

    # --- Save Model ---
    output_filename = f"{Path(model_path).stem}_tactics_trained.pth.tar"
    output_path = paths.checkpoints_dir / output_filename
    
    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # For potential continuation
        'epoch': args.epochs,
        'final_loss': avg_epoch_loss
    }, output_path)
    
    print(f"\n✅ Training complete. Saved updated model to:\n{output_path}")

if __name__ == '__main__':
    main()