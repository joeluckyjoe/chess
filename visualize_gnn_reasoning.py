# FILE: visualize_gnn_reasoning.py (Final Universal Version)
import argparse
import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Batch

# Assuming the project is run from the root directory
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input, CNN_INPUT_CHANNELS
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size

def load_model(checkpoint_path, device):
    """
    Loads the model from a checkpoint file, handling both full training checkpoints
    and raw pre-trained state_dict files.
    """
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
    
    model = ValueNextStateModel(**model_params)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # --- THIS IS THE FIX ---
    # Handle both types of checkpoint files
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Detected a full training checkpoint. Loading 'model_state_dict'.")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Detected a raw state_dict file. Loading directly.")
        model.load_state_dict(checkpoint) # Assumes the file is the state_dict
    
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} and set to evaluation mode.")
    return model

def get_board_at_move(pgn_path, move_number):
    """ Parses a PGN file and returns the board state *before* a specific move number. """
    with open(pgn_path) as pgn_file:
        game = chess.pgn.read_game(pgn_file)
    if not game:
        return None, None
    
    node = game
    for _ in range(move_number - 1):
        if node.next():
            node = node.next()
        else:
            return None, None
    
    return node.board(), node.next().move if node.next() else None

def visualize_reasoning(model, board, move, device, move_number):
    """ Main function to perform model inference and generate visualizations. """
    if move is None:
        print(f"Error: Could not find move {move_number} in PGN.")
        return
        
    print(f"\n--- Analyzing Board State Before Move {move_number}: {board.san(move)} ---")
    print(f"--- Board FEN: {board.fen()} ---")

    gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
    gnn_batch = Batch.from_data_list([gnn_data])
    cnn_tensor_batch = cnn_tensor.unsqueeze(0)

    with torch.no_grad():
        _, _, _, gnn_node_embeddings = model(gnn_batch, cnn_tensor_batch, return_embeddings=True)

    node_importance = torch.norm(gnn_node_embeddings, p=2, dim=1).cpu().numpy()
    
    if node_importance.max() > 0:
        normalized_importance = node_importance / node_importance.max()
    else:
        normalized_importance = np.zeros_like(node_importance)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')

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
    output_filename = f"analysis_move_{move_number-1}_before_{board.san(move)}_PRETRAINED.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nPre-trained model visualization saved to {output_filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize GNN reasoning for a specific chess move.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument('--pgn', type=str, required=True, help="Path to the PGN file of the game to analyze.")
    parser.add_argument('--move', type=int, required=True, help="The (half) move number in the PGN to analyze.")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)
    board, move = get_board_at_move(args.pgn, args.move)
    
    if board and move:
        visualize_reasoning(model, board, move, device, args.move)
    else:
        print(f"Could not reach move number {args.move} in the PGN file.")

if __name__ == "__main__":
    main()