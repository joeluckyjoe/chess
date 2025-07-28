import argparse
import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Batch

# Assuming the project is run from the root directory
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input, CNN_INPUT_CHANNELS, PIECE_FEATURE_DIM

def load_model(checkpoint_path, device):
    """ Loads the model from a .pth.tar checkpoint file. """
    # Parameters based on the provided instantiation snippet
    # NOTE: You may need to find the exact values for HIDDEN_DIM, EMBED_DIM, etc.
    # from your config file. These are common placeholder values.
    model_params = {
        'gnn_hidden_dim': 128,
        'cnn_in_channels': CNN_INPUT_CHANNELS,
        'embed_dim': 128,
        'policy_size': 4672,  # A common value for chess, confirm if different
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
    # Check for 'state_dict' key, common in training frameworks
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint) # If the checkpoint is just the state_dict

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
    for i, move in enumerate(game.mainline_moves()):
        if i + 1 == move_number:
            return node.board(), move
        node = node.next()
    return None, None


def visualize_reasoning(model, board, move, device, move_number):
    """ Main function to perform model inference and generate visualizations. """
    print(f"\n--- Analyzing Board State Before Move {move_number}: {move} ---")
    print(f"--- Board FEN: {board.fen()} ---")

    # 1. Convert board state to model input
    gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
    # The converter gives a single graph, but the model expects a batch.
    gnn_batch = Batch.from_data_list([gnn_data])
    # Also add a batch dimension to the CNN tensor
    cnn_tensor_batch = cnn_tensor.unsqueeze(0)

    # 2. Perform a forward pass
    with torch.no_grad():
        _, _, _, gnn_node_embeddings = model(gnn_batch, cnn_tensor_batch, return_embeddings=True)

    # 3. Visualize GNN Node Importance
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
            ax.text(file + 0.5, rank + 0.5, piece.unicode_symbol(), 
                    ha='center', va='center', fontsize=28, zorder=2)

    ax.set_xticks(np.arange(8) + 0.5, labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticks(np.arange(8) + 0.5, labels=list('87654321'))
    
    plt.title(f"GNN Node Importance Before Move {move_number}: {move}", fontsize=16)
    output_filename = f"analysis_move_{move_number-1}_before_{move}.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nVisualization saved to {output_filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize GNN reasoning for a specific chess move.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint (.pth.tar).")
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