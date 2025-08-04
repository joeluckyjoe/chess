# visualization/visualize_gnn_reasoning.py

import argparse
import chess
import chess.pgn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Batch
import sys
from pathlib import Path

# Add project root to Python's path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.neural_network.temporal_model import TemporalPolicyValueModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size
from config import config_params
from hardware_setup import get_device

GNN_METADATA = (
    ['square', 'piece'],
    [('square', 'adjacent_to', 'square'), ('piece', 'occupies', 'square'),
     ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')]
)

def load_model(checkpoint_path_str: str, device: torch.device) -> nn.Module:
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    is_temporal = any(key.startswith('transformer_encoder') for key in model_state_dict.keys())

    if is_temporal:
        print("Detected Temporal Model.")
        encoder_model = PolicyValueModel(
            gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
            embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
            gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
        )
        model = TemporalPolicyValueModel(
            encoder_model=encoder_model,
            policy_size=get_action_space_size(),
            d_model=config_params['EMBED_DIM']
        ).to(device)
    else:
        print("Detected GNN Model.")
        model = PolicyValueModel(
            gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'], cnn_in_channels=14,
            embed_dim=config_params['EMBED_DIM'], policy_size=get_action_space_size(),
            gnn_num_heads=config_params['GNN_NUM_HEADS'], gnn_metadata=GNN_METADATA
        ).to(device)
        
    model.load_state_dict(model_state_dict)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    return model

def get_board_from_pgn(pgn_path, move_number):
    with open(pgn_path) as pgn_file:
        game = chess.pgn.read_game(pgn_file)
    if not game:
        return None, None, None, None

    # We need to find the node corresponding to the position *before* the move number
    # A full move number (e.g., 5) corresponds to 2 plies (half-moves)
    target_ply = (move_number - 1) * 2
    
    node = game
    for i in range(target_ply):
        if node.is_end(): return None, None, None, None
        node = node.next()
    
    board = node.board()
    
    if node.is_end():
        return None, None, None, None

    move = node.next().move
    move_san = board.san(move)
    title = f"GNN Node Importance\nGame: {Path(pgn_path).name}, Before Move {move_number}: {move_san}"
    filename_suffix = f"gm{game.headers.get('Round', 'NA')}_mv{move_number}"
    return board, title, filename_suffix, move

def visualize_reasoning(model, board, device, title, checkpoint_path, filename_suffix):
    print(f"\n--- Analyzing Board State ---")
    print(f"--- FEN: {board.fen()} ---")

    gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
    gnn_batch = Batch.from_data_list([gnn_data])
    cnn_tensor_batch = cnn_tensor.unsqueeze(0)

    with torch.no_grad():
        model_to_call = model.encoder if isinstance(model, TemporalPolicyValueModel) else model
        _, _, gnn_node_embeddings = model_to_call(gnn_batch, cnn_tensor_batch, return_embeddings=True)

    node_importance = torch.norm(gnn_node_embeddings, p=2, dim=1).cpu().numpy()
    
    if node_importance.max() == 0:
        print("Warning: GNN node embeddings are all zero.")
        normalized_importance = np.zeros_like(node_importance)
    else:
        normalized_importance = node_importance / node_importance.max()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect('equal')

    for i in range(8):
        for j in range(8):
            color = 'white' if (i + j) % 2 == 0 else '#d3d3d3'
            ax.add_patch(plt.Rectangle((i, j), 1, 1, color=color, zorder=0))

    for square_index in range(64):
        rank = 7 - (square_index // 8)
        file = square_index % 8
        importance = normalized_importance[square_index]
        
        if importance > 0.01:
            circle = plt.Circle((file + 0.5, rank + 0.5), 
                                radius=importance * 0.45, color='#c23b22', 
                                alpha=max(0.1, importance * 0.7), zorder=1)
            ax.add_patch(circle)
            
        piece = board.piece_at(square_index)
        if piece:
            ax.text(file + 0.5, rank + 0.5, piece.unicode_symbol(invert_color=True), 
                    ha='center', va='center', fontsize=32, zorder=2)

    ax.set_xticks(np.arange(8) + 0.5, labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticks(np.arange(8) + 0.5, labels=list('87654321'))
    
    plt.title(title, fontsize=14)
    
    checkpoint_name = Path(checkpoint_path).stem
    output_dir = Path(__file__).resolve().parent
    output_filename = output_dir / f"analysis_{checkpoint_name}_{filename_suffix}.png"
    
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nVisualization saved to {output_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize GNN reasoning for a specific chess position.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fen', type=str, help="FEN string of the board position to analyze.")
    group.add_argument('--pgn', type=str, help="Path to the PGN file of the game to analyze.")
    parser.add_argument('--move-number', type=int, help="The full move number in the PGN to analyze (requires --pgn).")
    
    args = parser.parse_args()
    
    if args.pgn and args.move_number is None:
        parser.error("--move-number is required when using --pgn.")

    device = get_device()
    print(f"Using device: {device}")

    try:
        model = load_model(args.checkpoint, device)
        
        if args.fen:
            board = chess.Board(args.fen)
            title = f"GNN Node Importance\nFEN: {args.fen}"
            filename_suffix = "from_fen"
        else:
            board, title, filename_suffix, _ = get_board_from_pgn(args.pgn, args.move_number)
            if board is None:
                raise ValueError(f"Could not reach move number {args.move_number} in the PGN file.")
    
        visualize_reasoning(model, board, device, title, args.checkpoint, filename_suffix)

    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()