import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim
import chess
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Add the project root to the Python path
# This allows us to import from the gnn_agent package
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Project-specific Imports ---
from config import get_paths, config_params as default_config_params
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters import gnn_data_converter
from gnn_agent.gamestate_converters.action_space_converter import move_to_index, get_action_space_size

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

def load_model_from_checkpoint(model_path: Path, device: torch.device) -> ChessNetwork:
    """
    Loads a ChessNetwork model from a checkpoint, correctly building the
    network from the configuration saved within the checkpoint file.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    
    model_config = checkpoint.get('config_params', default_config_params)
    
    print("Building network from checkpoint configuration...")
    square_gnn = SquareGNN(
        in_features=model_config.get('SQUARE_IN_FEATURES', 12),
        hidden_features=model_config.get('SQUARE_HIDDEN_FEATURES', 256),
        out_features=model_config.get('SQUARE_OUT_FEATURES', 128),
        heads=model_config.get('SQUARE_ATTENTION_HEADS', 4)
    )
    piece_gnn = PieceGNN(
        in_channels=model_config.get('PIECE_IN_CHANNELS', 12),
        hidden_channels=model_config.get('PIECE_HIDDEN_CHANNELS', 256),
        out_channels=model_config.get('PIECE_OUT_CHANNELS', 128)
    )
    cross_attention = CrossAttentionModule(
        sq_embed_dim=model_config.get('SQUARE_OUT_FEATURES', 128),
        pc_embed_dim=model_config.get('PIECE_OUT_CHANNELS', 128),
        num_heads=model_config.get('CROSS_ATTENTION_HEADS', 4)
    )
    policy_head = PolicyHead(
        embedding_dim=model_config.get('SQUARE_OUT_FEATURES', 128),
        num_possible_moves=get_action_space_size()
    )
    value_head = ValueHead(embedding_dim=model_config.get('SQUARE_OUT_FEATURES', 128))
    
    network = ChessNetwork(
        square_gnn, 
        piece_gnn, 
        cross_attention, 
        policy_head, 
        value_head
    ).to(device)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    network.load_state_dict(state_dict)
    
    print(f"Successfully loaded model weights from {model_path.name}")
    return network

def collate_puzzles(batch: list[Dict[str, Any]]):
    """Custom collate function to process a batch of puzzles."""
    gnn_inputs = []
    policy_targets = []
    
    for puzzle in batch:
        fen = puzzle['fen']
        best_move_uci = puzzle['best_move']
        
        board = chess.Board(fen)
        move = chess.Move.from_uci(best_move_uci)

        gnn_input = gnn_data_converter.convert_to_gnn_input(board, device='cpu')
        
        try:
            # --- BUG FIX: Pass the 'board' argument to the move_to_index function ---
            move_idx = move_to_index(move, board)
            gnn_inputs.append(gnn_input)
            policy_targets.append(move_idx)
        except ValueError as e:
            print(f"Warning: Skipping move {best_move_uci} in FEN {fen}. Reason: {e}")

    if not policy_targets:
        return None, None

    return gnn_inputs, torch.tensor(policy_targets, dtype=torch.long)


def main():
    """Main function to run the tactical training session."""
    parser = argparse.ArgumentParser(description="Run supervised training on tactical puzzles.")
    parser.add_argument("--puzzles_path", type=str, required=True, help="Path to the tactical_puzzles.jsonl file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to a specific model checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    args = parser.parse_args()

    paths = get_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    try:
        network = load_model_from_checkpoint(Path(args.model_path), device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    network.train()

    # --- Load Data ---
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

    # --- Training Setup ---
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    policy_loss_fn = torch.nn.CrossEntropyLoss()

    # --- Training Loop ---
    for epoch in range(args.epochs):
        total_policy_loss = 0
        puzzles_processed = 0

        for gnn_inputs_batch, policy_targets_batch in data_loader:
            if gnn_inputs_batch is None:
                continue

            optimizer.zero_grad()
            
            policy_targets_batch = policy_targets_batch.to(device)
            
            batch_policy_loss = 0
            # Process each item in the batch individually
            # A more optimized version could truly batch the graph data, but this is safer.
            for i, gnn_input in enumerate(gnn_inputs_batch):
                # Move individual graph tensors to the correct device
                gnn_input_on_device = tuple(t.to(device) for t in gnn_input)

                policy_logits, _ = network(*gnn_input_on_device)
                
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
    model_path = Path(args.model_path)
    # Correctly handle suffixes like .pth.tar
    base_stem = model_path.name.replace('.pth.tar', '')
    output_filename = f"{base_stem}_tactics_trained.pth.tar"
    output_path = paths.checkpoints_dir / output_filename
    
    # We need to save the original config params back into the checkpoint
    original_checkpoint = torch.load(model_path, map_location='cpu')
    original_config = original_checkpoint.get('config_params', default_config_params)
    original_game_num = original_checkpoint.get('game_number', 0)

    torch.save({
        'game_number': original_game_num,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config_params': original_config,
        'epoch': args.epochs,
        'final_loss': avg_epoch_loss
    }, output_path)
    
    print(f"\n✅ Training complete. Saved updated model to:\n{output_path}")

if __name__ == '__main__':
    main()
