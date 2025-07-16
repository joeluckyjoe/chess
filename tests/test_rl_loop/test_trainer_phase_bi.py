#
# File: tests/test_rl_loop/test_trainer_phase_bi.py (Corrected)
#
import unittest
from unittest.mock import MagicMock, patch, ANY
import torch
import chess
from pathlib import Path

# We need the real converter and network for this integration-style test
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input

class TestTrainerPhaseBI(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model_config = {
            'GNN_EMBED_DIM': 32,
            'CNN_EMBED_DIM': 32,
            'GNN_HIDDEN_DIM': 32,
            'NUM_HEADS': 2,
            'LEARNING_RATE': 1e-4,
            'WEIGHT_DECAY': 1e-4,
            'VALUE_LOSS_WEIGHT': 1.0,
            'MATERIAL_BALANCE_LOSS_WEIGHT': 0.5 # Test value
        }
        self.trainer = Trainer(self.model_config, self.device)
        
        # We still need to mock the network itself for predictable outputs
        # but we let the data converter run for real.
        self.trainer.network = MagicMock(spec=ChessNetwork)
        
        # --- FIX: Create a real parameter that the optimizer can track ---
        # This parameter will be used to create a valid computation graph.
        self.mock_param = torch.nn.Parameter(torch.tensor(1.0))
        self.trainer.network.parameters.return_value = [self.mock_param]
        # --- END FIX ---
        
        self.trainer.optimizer = torch.optim.Adam(self.trainer.network.parameters(), lr=1e-4)

    def test_train_on_batch_loss_calculation(self):
        """
        Tests that all three loss components (policy, value, material) are calculated
        and combined correctly. This test now uses the real data converter.
        """
        # --- Mock Network Outputs ---
        # The network forward pass is the only thing we need to mock now.
        pred_policy = torch.randn(1, 4672, requires_grad=True)
        
        # --- FIX: Multiply by the mock parameter to connect the computation graph ---
        # This ensures that when loss.backward() is called, a gradient will flow to our mock_param.
        pred_value = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True) * self.mock_param
        pred_material = torch.tensor([[1.8]], dtype=torch.float32, requires_grad=True) * self.mock_param
        # --- END FIX ---

        self.trainer.network.return_value = (pred_policy, pred_value, pred_material)
        
        # --- Real Test Data ---
        # CORRECTED FEN: White is up a pawn (black d7-pawn is missing).
        # The true material balance should be +1.0 from white's perspective.
        board = chess.Board("rnbqkbnr/ppppp1pp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        game_examples = [(board.fen(), {chess.Move.from_uci("g1f3"): 1.0}, 1.0)] # White wins (outcome = 1.0)
        puzzle_examples = []
        
        # --- Execute ---
        avg_p_loss, avg_v_loss, avg_m_loss = self.trainer.train_on_batch(
            game_examples, puzzle_examples, batch_size=1
        )

        # --- Assertions ---
        # 1. Check that the network was called
        self.trainer.network.assert_called_once()
        
        # 2. Check that the losses are reasonable floats
        self.assertIsInstance(avg_p_loss, float)
        self.assertIsInstance(avg_v_loss, float)
        self.assertIsInstance(avg_m_loss, float)
        self.assertGreater(avg_p_loss, 0)
        self.assertGreater(avg_v_loss, 0)
        self.assertGreater(avg_m_loss, 0)

        # 3. Manually calculate expected losses for this single data point
        # Value loss: (predicted_value - target_outcome)^2 = (0.5 - 1.0)^2 = 0.25
        expected_v_loss = (0.5 - 1.0)**2
        self.assertAlmostEqual(avg_v_loss, expected_v_loss, places=5)
        
        # Material loss: (predicted_material - target_material)^2 = (1.8 - 1.0)^2 = 0.64
        expected_m_loss = (1.8 - 1.0)**2
        self.assertAlmostEqual(avg_m_loss, expected_m_loss, places=5)
        
        # 4. Check if backward() was called on a combined loss by checking for gradients
        self.assertIsNotNone(self.mock_param.grad)


if __name__ == '__main__':
    unittest.main()
