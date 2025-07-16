#
# File: tests/gamestate_converters/test_gnn_data_converter_phase_bi.py
#
import unittest
import torch
import chess
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input, PIECE_MATERIAL_VALUE

class TestGNNDataConverterPhaseBI(unittest.TestCase):
    def setUp(self):
        """Set up a device to run tests on."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on {self.device}")

    def test_return_type_and_shape(self):
        """
        Tests that the converter returns three items and the material balance
        tensor has the correct shape and type.
        """
        board = chess.Board()
        outputs = convert_to_gnn_input(board, self.device)

        # 1. Check for three return values
        self.assertEqual(len(outputs), 3, "Function should return three values.")

        gnn_data, cnn_data, material_balance = outputs

        # 2. Check the type and shape of the material balance tensor
        self.assertIsInstance(material_balance, torch.Tensor)
        self.assertEqual(material_balance.shape, (1,), "Material balance tensor should have shape (1,).")
        self.assertEqual(material_balance.dtype, torch.float32, "Material balance should be a float32 tensor.")

    def test_material_balance_calculation(self):
        """
        Tests the perspective-based material balance calculation in various scenarios.
        """
        # Scenario 1: Starting position (equal material)
        board = chess.Board()
        _, _, material_balance = convert_to_gnn_input(board, self.device)
        self.assertAlmostEqual(material_balance.item(), 0.0, "Balance should be 0 for the starting position.")

        # Scenario 2: White is up a pawn, White's turn
        board = chess.Board()
        board.remove_piece_at(chess.D7) # Remove black d-pawn
        board.turn = chess.WHITE
        expected_balance = PIECE_MATERIAL_VALUE[chess.PAWN]
        _, _, material_balance = convert_to_gnn_input(board, self.device)
        self.assertAlmostEqual(material_balance.item(), expected_balance, "White should be up by a pawn's value.")

        # Scenario 3: White is up a pawn, Black's turn
        board.turn = chess.BLACK
        expected_balance_perspective = -PIECE_MATERIAL_VALUE[chess.PAWN]
        _, _, material_balance = convert_to_gnn_input(board, self.device)
        self.assertAlmostEqual(material_balance.item(), expected_balance_perspective, "Perspective balance for Black should be negative.")

        # Scenario 4: Black is up a rook, Black's turn
        board = chess.Board()
        board.remove_piece_at(chess.A1) # Remove white a-rook
        board.turn = chess.BLACK
        expected_balance = PIECE_MATERIAL_VALUE[chess.ROOK]
        _, _, material_balance = convert_to_gnn_input(board, self.device)
        self.assertAlmostEqual(material_balance.item(), expected_balance, "Black should be up by a rook's value.")
        
        # Scenario 5: Black is up a rook, White's turn
        board.turn = chess.WHITE
        expected_balance_perspective = -PIECE_MATERIAL_VALUE[chess.ROOK]
        _, _, material_balance = convert_to_gnn_input(board, self.device)
        self.assertAlmostEqual(material_balance.item(), expected_balance_perspective, "Perspective balance for White should be negative.")


if __name__ == '__main__':
    unittest.main()