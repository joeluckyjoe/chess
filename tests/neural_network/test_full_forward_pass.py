#
# File: tests/neural_network/test_full_forward_pass.py (Corrected for function-based converter)
#
import unittest
import torch
import chess
from torch_geometric.data import Batch

from gnn_agent.neural_network.chess_network import ChessNetwork
# --- CORRECTED IMPORT ---
# Import the function, not a class that doesn't exist.
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input

class TestUnifiedNetworkForwardPass(unittest.TestCase):
    """
    Tests the full forward pass of the modern ChessNetwork. This version is corrected
    to use the project's function-based game state converter.
    """
    def setUp(self):
        """Set up the modern ChessNetwork."""
        self.model = ChessNetwork(embed_dim=64, gnn_hidden_dim=32, num_heads=2)
        self.model.eval()  # Set the model to evaluation mode
        # --- REMOVED OBSOLETE LINE ---
        # No converter class to instantiate.

    def test_forward_pass_initial_position(self):
        """
        Tests a full forward pass starting from the initial board position.
        This validates the integration of the GATv2Conv-based UnifiedGNN.
        """
        # 1. Create a board and get the model's device
        board = chess.Board()
        device = next(self.model.parameters()).device

        # 2. Convert board state using the imported function
        # --- CORRECTED CALL ---
        data = convert_to_gnn_input(board, device)

        # 3. Create a batch of size 1, as the model expects
        batch = Batch.from_data_list([data])

        # 4. Perform the forward pass
        with torch.no_grad():
            policy_logits, value = self.model(batch)

        # 5. Check the output shapes and value range
        self.assertIsInstance(policy_logits, torch.Tensor)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(policy_logits.shape, (1, 4672))
        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(-1.0 <= value.item() <= 1.0)
        
    def test_forward_pass_empty_board(self):
        """
        Tests the forward pass on an empty board edge case.
        """
        # 1. Create an empty board and get device
        board = chess.Board(fen=None)
        device = next(self.model.parameters()).device

        # 2. Convert using the function
        # --- CORRECTED CALL ---
        data = convert_to_gnn_input(board, device)
        
        # 3. Create a batch
        batch = Batch.from_data_list([data])
        
        # 4. Perform the forward pass
        with torch.no_grad():
            policy_logits, value = self.model(batch)
        
        # 5. Check outputs
        self.assertEqual(policy_logits.shape, (1, 4672))
        self.assertEqual(value.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()