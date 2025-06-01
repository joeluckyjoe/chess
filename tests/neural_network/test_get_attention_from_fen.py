# tests/test_neural_network/test_get_attention_from_fen.py

import torch
import chess
import matplotlib.pyplot as plt
import numpy as np
import unittest

# --- Real Project Imports ---
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead

# --- Plotting Functions (Unchanged) ---

def plot_square_attention_to_pieces(
    attention_weights,
    square_name,
    piece_labels,
    ax=None
):
    """Visualizes attention from one square to all pieces as a bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(piece_labels))
    ax.barh(y_pos, attention_weights, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(piece_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Attention Weight")
    ax.set_title(f"Attention from Square {square_name} to All Pieces")
    plt.tight_layout()
    return ax

def plot_all_squares_attention_to_piece(
    attention_weights,
    piece_label,
    piece_index,
    ax=None
):
    """Visualizes attention from all 64 squares to one piece as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    heatmap_data = attention_weights[:, piece_index].reshape(8, 8) # Assuming square-major order for attention weights
    
    im = ax.imshow(heatmap_data, cmap="hot", interpolation="nearest")
    
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticklabels(range(8, 0, -1)) # Chessboard ranks are 8 to 1
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")

    ax.set_title(f"Attention from All Squares to Piece: {piece_label} (Index {piece_index})")
    return ax

# --- Main Test and Visualization Class ---

class TestAttentionVisualization(unittest.TestCase):

    def test_get_and_plot_attention_from_fen(self):
        """
        Tests the full pipeline from FEN -> Converter -> Network -> Attention -> Plots.
        """
        # --- 1. Setup ---
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1" # e.g., after 1. e4 e5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        board = chess.Board(fen)
        gnn_input, piece_labels_for_plot = convert_to_gnn_input(board, device)
        
        # --- 2. Initialize the REAL Network (with mock weights) ---
        
        square_gnn = SquareGNN(
            in_features=12,       # From GNNDataConverter square features
            hidden_features=64,   # Intermediate GNN layer size
            out_features=128      # Output embedding size for squares
        )
        piece_gnn = PieceGNN(
            in_channels=12,       # From GNNDataConverter piece features
            hidden_channels=64,   # Intermediate GNN layer size
            out_channels=128      # Output embedding size for pieces
        )
        
        cross_attention = CrossAttentionModule(
            sq_embed_dim=128,     # Output dim of SquareGNN
            pc_embed_dim=128,     # Output dim of PieceGNN
            num_heads=4
        )
        
        # *** CORRECTED PolicyHead and ValueHead INSTANTIATION ***
        policy_head = PolicyHead(
            embedding_dim=128     # Input dim from ChessNetwork's processed embeddings
        )
        value_head = ValueHead(
            embedding_dim=128     # Input dim from ChessNetwork's processed embeddings
        )
        # *** END CORRECTION ***

        model = ChessNetwork(
            square_gnn, piece_gnn, cross_attention, policy_head, value_head
        ).to(device)
        model.eval() # Set to evaluation mode

        # --- 3. Get Attention Weights ---
        # The GNNInput needs to be "batched" for the network, even if batch size is 1.
        # The network expects inputs like (batch_size, num_nodes, features) or (num_nodes, batch_size, features)
        # depending on batch_first conventions.
        # Our GNNDataConverter currently returns unbatched tensors.
        # The ChessNetwork's forward method expects (batch_size, num_squares, sq_embed_dim) for the policy/value heads
        # and the CrossAttentionModule expects (num_squares, batch_size, sq_embed_dim) and (num_pieces, batch_size, pc_embed_dim)
        # Let's ensure the inputs to the model are shaped correctly for a single batch item.
        # Square graph features: (num_squares, features) -> (1, num_squares, features) or (num_squares, 1, features)
        # Piece graph features: (num_pieces, features) -> (1, num_pieces, features) or (num_pieces, 1, features)
        
        # For this test, ChessNetwork's forward method handles the batching internally if inputs are unbatched.
        # However, the attention weights returned by CrossAttentionModule are (batch_size, num_squares, num_pieces)
        # or (num_squares, num_pieces) if batch_size is 1 and squeezed.
        # Let's assume the ChessNetwork forward pass handles this and returns attention weights
        # as (num_squares, num_pieces) for a single instance after potential unsqueezing/squeezing.

        with torch.no_grad():
            # Perform a forward pass, telling the model we want the attention weights
            # The ChessNetwork's forward method should handle the batch dimension internally for a single FEN.
            # It should return policy_logits, value, and attention_weights (if requested)
            # attention_weights from CrossAttentionModule: (batch_size, num_query_elements, num_key_elements)
            # For us: (1, 64, num_pieces)
            _, _, attention_weights_batch = model( # Expecting batched output
                gnn_input.square_graph.x,
                gnn_input.square_graph.edge_index,
                gnn_input.piece_graph.x,
                gnn_input.piece_graph.edge_index,
                gnn_input.piece_to_square_map,
                return_attention=True  # The flag to get attention weights
            )
        
        # We expect attention_weights_batch to be (1, 64, num_pieces)
        # For plotting, we need (64, num_pieces)
        self.assertIsNotNone(attention_weights_batch, "Attention weights should not be None")
        self.assertEqual(attention_weights_batch.ndim, 3, "Attention weights should be 3D (batch, query, key)")
        self.assertEqual(attention_weights_batch.shape[0], 1, "Batch dimension should be 1")
        
        attention_weights = attention_weights_batch.squeeze(0).cpu().numpy() # Remove batch dim, move to CPU & NumPy
        
        # Shape is (num_query_elements, num_key_elements) which is (squares, pieces) -> (64, num_pieces)
        
        self.assertIsNotNone(attention_weights)
        self.assertEqual(attention_weights.shape[0], 64, "Attention weights should have 64 rows (squares)") 
        self.assertEqual(attention_weights.shape[1], len(piece_labels_for_plot), "Attention weights columns should match num_pieces")

        print(f"\nSuccessfully retrieved attention weights with shape: {attention_weights.shape}")
        print(f"Piece labels for plotting ({len(piece_labels_for_plot)} total): {piece_labels_for_plot}")

        # --- 4. Generate Plots ---
        square_to_plot = 'e4' # White's pawn
        # Let's pick a specific piece that exists on the board for the second plot.
        # For FEN "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        # Black pawn on e5 is a good candidate.
        # piece_labels_for_plot is like: ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'r', 'n', 'b', 'q', 'k', 'b', 'n', 'r', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p']
        # The black pawn on e5 should be 'p'. Let's find its index.
        
        square_index_to_plot = chess.parse_square(square_to_plot) # e.g., 28 for e4

        # Find the black pawn on e5 (square index 36)
        target_piece_symbol_for_plot2 = 'p' # black pawn
        target_piece_square_for_plot2 = chess.E5 # square e5
        
        # Find the index of this specific piece in our piece_labels_for_plot
        # piece_labels_for_plot is ordered based on board.piece_map().keys()
        # gnn_input.piece_to_square_map maps piece graph indices to board square indices
        
        piece_index_for_plot2 = -1
        for i, sq_idx_tensor in enumerate(gnn_input.piece_to_square_map):
            sq_idx = sq_idx_tensor.item()
            if sq_idx == target_piece_square_for_plot2 and piece_labels_for_plot[i] == target_piece_symbol_for_plot2:
                piece_index_for_plot2 = i
                break
        
        if piece_index_for_plot2 == -1:
             self.fail(f"Could not find piece '{target_piece_symbol_for_plot2}' on square {chess.square_name(target_piece_square_for_plot2)} in piece_labels_for_plot.")


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Bar chart for one square's attention to all pieces
        plot_square_attention_to_pieces(
            attention_weights[square_index_to_plot, :], # Attention from e4 to all pieces
            square_to_plot,
            piece_labels_for_plot,
            ax=ax1
        )
        
        # Plot 2: Heatmap of all squares' attention to one piece (black pawn on e5)
        plot_all_squares_attention_to_piece(
            attention_weights, # Full attention matrix (64, num_pieces)
            f"{target_piece_symbol_for_plot2} on {chess.square_name(target_piece_square_for_plot2)}",
            piece_index_for_plot2, # Index of the black pawn on e5 in the piece_graph
            ax=ax2
        )

        plt.suptitle(f"Cross-Attention Visualization for FEN: {fen}", fontsize=16)
        output_filename = "attention_visualization_real_converter_and_network.png" # New name
        plt.savefig(output_filename)
        print(f"Generated visualization and saved to {output_filename}")
        plt.close(fig) # Close the figure to prevent it from displaying in interactive environments

if __name__ == '__main__':
    unittest.main()