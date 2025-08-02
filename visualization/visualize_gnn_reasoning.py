import argparse
import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Batch
import sys
from pathlib import Path

# Add project root to Python's path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input, CNN_INPUT_CHANNELS
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size

def load_model(checkpoint_path, device):
    """Loads the PolicyValueModel from a checkpoint file."""
    model_params = {
        'gnn_hidden_dim': 128,
        'cnn_in_channels': CNN_INPUT_CHANNELS,
        'embed_dim': 256,
        'policy_size': get_action_space_size(),
        'gnn_num_heads': 4,
        'gnn_metadata': (
            ['square', 'piece'],
            [
                ('square', 'adjacent_to', 'square'),
                ('piece', 'occupies', 'square'),
                ('piece', 'attacks', 'piece'),
                ('piece', 'defends', 'piece')
            ]
        )
    }
    
    model = PolicyValueModel(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Detected a full training checkpoint. Loading 'model_state_dict'.")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Detected a raw state_dict file. Loading directly.")
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} and set to evaluation mode.")
    return model

def get_board_at_move(pgn_path, move_number):
    """Parses a PGN file and returns the board state *before* a specific move number."""
    with open(pgn_path) as pgn_file:
        game = chess.pgn.read_game(pgn_file)
    if not game:
        return None, None, None
    
    node = game
    for _ in range(move_number - 1):
        if node.next():
            node = node.next()
        else:
            return None, None, None
    
    return game, node.board(), node.next().move if node.next() else None

# --- FIX: Added 'game' object to the function signature ---
def visualize_reasoning(model, board, move, device, move_number, checkpoint_path, game):
    """Main function to perform model inference and generate visualizations."""
    if move is None:
        print(f"Error: Could not find move {move_number} in PGN.")
        return
        
    print(f"\n--- Analyzing Board State Before Move {move_number}: {board.san(move)} ---")
    print(f"--- Board FEN: {board.fen()} ---")

    gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
    gnn_batch = Batch.from_data_list([gnn_data])
    cnn_tensor_batch = cnn_tensor.unsqueeze(0)

    with torch.no_grad():
        _, _, gnn_node_embeddings = model(gnn_batch, cnn_tensor_batch, return_embeddings=True)

    node_importance = torch.norm(gnn_node_embeddings, p=2, dim=1).cpu().numpy()
    
    if node_importance.max() > 0:
        normalized_importance = node_importance / node_importance.max()
    else:
        normalized_importance = np.zeros_like(node_importance)

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
                                radius=importance * 0.45,
                                color='#c23b22', 
                                alpha=max(0.1, importance * 0.7),
                                zorder=1)
            ax.add_patch(circle)
            
        piece = board.piece_at(square_index)
        if piece:
            ax.text(file + 0.5, rank + 0.5, piece.unicode_symbol(invert_color=True), 
                    ha='center', va='center', fontsize=32, zorder=2)

    ax.set_xticks(np.arange(8) + 0.5, labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticks(np.arange(8) + 0.5, labels=list('87654321'))
    
    plt.title(f"GNN Node Importance Before Move {move_number}: {board.san(move)}", fontsize=16)
    
    checkpoint_name = Path(checkpoint_path).stem
    san_move_str = board.san(move).replace('+', 'x').replace('#', 'm')
    
    output_dir = Path(__file__).resolve().parent
    output_filename = output_dir / f"analysis_{checkpoint_name}_gm{game.headers.get('Round', 'N_A')}_mv{move_number}.png"
    
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nVisualization saved to {output_filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize GNN reasoning for a specific chess move.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument('--pgn', type=str, required=True, help="Path to the PGN file of the game to analyze.")
    parser.add_argument('--move-number', type=int, required=True, help="The (half) move number in the PGN to analyze.")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = load_model(args.checkpoint, device)
        game, board, move = get_board_at_move(args.pgn, args.move_number)
    
        if board and move:
            # --- FIX: Pass the 'game' object to the visualization function ---
            visualize_reasoning(model, board, move, device, args.move_number, args.checkpoint, game)
        else:
            print(f"Could not reach move number {args.move_number} in the PGN file.")
    except FileNotFoundError as e:
        print(f"ERROR: File not found. Please check your paths. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()