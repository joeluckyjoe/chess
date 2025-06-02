#
# File: tests/test_neural_network/test_get_attention_from_fen.py (Final)
#
import unittest
import torch
import chess
import matplotlib.pyplot as plt
import numpy as np
import os

from gnn_agent.gamestate_converters import gnn_data_converter
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead

# Configuration
GNN_INPUT_FEATURES = 12
GNN_OUTPUT_FEATURES = 128
NUM_ATTENTION_HEADS = 4
POLICY_HEAD_MOVE_CANDIDATES = 4672

def plot_attention_weights(attention_weights, piece_labels, square_labels, fen, save_path=None):
    """Plots the final attention weights (averaged over heads)."""
    if attention_weights is None:
        print("Attention weights are None, cannot plot.")
        return
    
    weights = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(weights, cmap='viridis')

    ax.set_xticks(np.arange(len(square_labels)))
    ax.set_yticks(np.arange(len(piece_labels)))
    ax.set_xticklabels(square_labels)
    ax.set_yticklabels(piece_labels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    ax.set_title(f"Cross-Attention (Pieces to Squares) for FEN: {fen}")
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Attention plot saved to {save_path}")
    else:
        plt.show()

def get_attention_from_fen(fen_string, network, device):
    """Takes a FEN, processes it, and returns attention weights and labels."""
    board = chess.Board(fen_string)
    gnn_input_data, piece_labels = gnn_data_converter.convert_to_gnn_input(
        board, device=device, for_visualization=True
    )
    with torch.no_grad():
        policy_logits, value_logit, attention_weights = network(*gnn_input_data, return_attention=True)
    return attention_weights, piece_labels

class TestAttentionVisualization(unittest.TestCase):
    def test_visualize_attention_from_start_pos(self):
        """An integration test to visualize attention from the starting position."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        square_gnn = SquareGNN(GNN_INPUT_FEATURES, GNN_OUTPUT_FEATURES, GNN_OUTPUT_FEATURES)
        piece_gnn = PieceGNN(GNN_INPUT_FEATURES, GNN_OUTPUT_FEATURES, GNN_OUTPUT_FEATURES)
        attention_module = CrossAttentionModule(
            sq_embed_dim=GNN_OUTPUT_FEATURES, pc_embed_dim=GNN_OUTPUT_FEATURES, num_heads=NUM_ATTENTION_HEADS
        )
        policy_head = PolicyHead(GNN_OUTPUT_FEATURES, POLICY_HEAD_MOVE_CANDIDATES)
        value_head = ValueHead(GNN_OUTPUT_FEATURES)

        network = ChessNetwork(
            square_gnn, piece_gnn, attention_module, policy_head, value_head
        ).to(device)
        network.eval()

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        attention_weights, piece_labels = get_attention_from_fen(fen, network, device)

        self.assertIsNotNone(attention_weights, "Attention weights should not be None.")
        
        # CORRECTED ASSERTIONS: Check for shape (num_pieces, num_squares)
        self.assertEqual(attention_weights.shape[0], len(piece_labels))
        self.assertEqual(attention_weights.shape[1], 64)

        print(f"Successfully retrieved attention weights of shape: {attention_weights.shape}")
        
        square_labels = [chess.square_name(sq) for sq in chess.SQUARES]
        output_dir = "test_outputs"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "attention_map_start_pos.png")

        plot_attention_weights(attention_weights, piece_labels, square_labels, fen, save_path=save_path)

if __name__ == '__main__':
    unittest.main()