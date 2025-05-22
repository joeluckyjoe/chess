# test_chess_environment.py
import unittest
from chess_environment import ChessEnvironment # Assuming your file is named chess_environment.py
import chess

class TestChessEnvironment(unittest.TestCase):

    def test_initial_board_state_fen(self):
        """
        Tests that the board initializes to the standard starting FEN.
        """
        env = ChessEnvironment()
        # Standard FEN for starting position
        expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.assertEqual(env.get_current_state_fen(), expected_fen)

    def test_initial_legal_moves(self):
        """
        Tests that the initial legal moves are correct (20 possible moves).
        """
        env = ChessEnvironment()
        legal_moves = env.get_legal_moves()
        self.assertEqual(len(legal_moves), 20)
        # Spot check a few common starting moves
        self.assertIn("g1f3", legal_moves)
        self.assertIn("e2e4", legal_moves)
        self.assertIn("d2d4", legal_moves)
        self.assertNotIn("e1g1", legal_moves) # Castling not legal yet from start

    def test_initial_player(self):
        """
        Tests that the initial player is White.
        """
        env = ChessEnvironment()
        self.assertTrue(env.get_current_player(), "Initial player should be White")

if __name__ == '__main__':
    unittest.main()