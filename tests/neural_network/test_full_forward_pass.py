#
# File: test_full_forward_pass.py
#
"""
Unit tests for the complete forward pass of the DualGNNAgent.
"""
import unittest
import torch
import chess
from gnn_data_converter import convert_to_gnn_input # GNNInput is also implicitly imported
from dual_gnn_agent import DualGNNAgent

class TestFullForwardPass(unittest.TestCase):
    
    def test_forward_pass_shapes_and_execution(self):
        """
        Tests a full forward pass of the DualGNNAgent model.
        
        This test ensures that:
        1. The model can be instantiated with new parameters.
        2. It can process GNN input derived from a real board state.
        3. The forward pass executes without runtime errors.
        4. The policy and value heads produce outputs of the correct shape.
        """
        # 1. Model Configuration
        gnn_input_dim = 12      # As per Global Plan (e.g., 12 features for squares/pieces)
        gnn_hidden_dim = 32     # Example hidden dimension for GNN layers
        gnn_embedding_dim = 64  # Example output embedding dimension from GNNs
        square_gnn_heads = 4    # Number of heads for SquareGNN's GAT layers
        n_heads_ca = 4          # Example number of heads for Cross-Attention
        num_possible_moves = 4672
        
        # 2. Instantiate the full agent
        agent = DualGNNAgent(
            gnn_input_dim=gnn_input_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_embedding_dim=gnn_embedding_dim,
            square_gnn_heads=square_gnn_heads,
            n_heads_cross_attention=n_heads_ca
        )
        agent.eval() # Set to evaluation mode

        # 3. Create realistic input data from the starting position
        board = chess.Board()
        gnn_input_data = convert_to_gnn_input(board) # Renamed to avoid conflict

        # 4. Perform the forward pass
        with torch.no_grad():
            try:
                policy_logits, value = agent(gnn_input_data)
            except Exception as e:
                self.fail(f"Forward pass failed with an exception: {e}")

        # 5. Check output shapes
        # Policy logits should be (batch_size, num_possible_moves)
        # Batch size is 1 for this test case
        self.assertEqual(policy_logits.shape, (1, num_possible_moves))
        
        # Value should be (batch_size, 1)
        self.assertEqual(value.shape, (1, 1))
        
        # 6. Check value range
        # The value output should be between -1 and 1 due to tanh
        self.assertTrue(-1.0 <= value.item() <= 1.0, f"Value {value.item()} out of range [-1, 1]")
        print("\nFull forward pass test successful.")
        print(f"Policy output shape: {policy_logits.shape}")
        print(f"Value output shape: {value.shape}")
        print(f"Sample value: {value.item():.4f}")

    def test_forward_pass_empty_board_pieces(self):
        """
        Tests the forward pass with a board state that might result in no pieces
        for the piece GNN (e.g., only kings left, or a custom FEN).
        This tests robustness to empty piece_graph.x.
        """
        gnn_input_dim = 12
        gnn_hidden_dim = 32
        gnn_embedding_dim = 64
        square_gnn_heads = 4
        n_heads_ca = 4
        num_possible_moves = 4672

        agent = DualGNNAgent(
            gnn_input_dim=gnn_input_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_embedding_dim=gnn_embedding_dim,
            square_gnn_heads=square_gnn_heads,
            n_heads_cross_attention=n_heads_ca
        )
        agent.eval()

        # Create a board with only kings (no other pieces for PieceGNN)
        # A more extreme case would be a board where convert_to_gnn_input
        # results in piece_graph.x being None or empty.
        # For this test, let's use a FEN that has very few pieces.
        # The GNNDataConverter should handle this and produce an empty piece_graph if needed.
        board = chess.Board("8/8/8/4k3/8/3K4/8/8 w - - 0 1") # Only kings
        gnn_input_data = convert_to_gnn_input(board)
        
        # Ensure piece_graph.x is indeed empty or has 0 pieces
        # This depends on gnn_data_converter's behavior.
        # If it creates nodes for kings, this test might not be "empty" in the strictest sense.
        # The crucial part is that PieceGNN and CrossAttention can handle it.
        print(f"\nTesting with board: {board.fen()}")
        if gnn_input_data.piece_graph.x is not None:
            print(f"Number of pieces for PieceGNN: {gnn_input_data.piece_graph.x.shape[0]}")
        else:
            print("Piece graph x is None")


        with torch.no_grad():
            try:
                policy_logits, value = agent(gnn_input_data)
            except Exception as e:
                self.fail(f"Forward pass with few/no pieces failed: {e}")

        self.assertEqual(policy_logits.shape, (1, num_possible_moves))
        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(-1.0 <= value.item() <= 1.0, f"Value {value.item()} out of range [-1, 1]")
        print("Forward pass test with few/no pieces successful.")
        print(f"Policy output shape: {policy_logits.shape}")
        print(f"Value output shape: {value.shape}")
        print(f"Sample value: {value.item():.4f}")


if __name__ == '__main__':
    unittest.main()