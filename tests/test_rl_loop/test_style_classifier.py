import unittest
import chess
from gnn_agent.rl_loop.style_classifier import StyleClassifier

class TestStyleClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = StyleClassifier()

    def test_pawn_push(self):
        """Tests that an aggressive pawn push is rewarded."""
        board = chess.Board("rnbqkbnr/p1pppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        move = chess.Move.from_uci("e4e5")
        self.assertAlmostEqual(self.classifier.score_move(board, move), self.classifier.pawn_push_reward)

    def test_capture(self):
        """Tests that a simple capture is rewarded."""
        board = chess.Board("rnbqkbnr/ppp2ppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
        move = chess.Move.from_uci("e4d5")
        self.assertAlmostEqual(self.classifier.score_move(board, move), self.classifier.piece_values[chess.PAWN])
        
    def test_en_passant_capture(self):
        """Tests that an en passant capture is rewarded."""
        board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        move = chess.Move.from_uci("e5f6")
        self.assertAlmostEqual(self.classifier.score_move(board, move), self.classifier.piece_values[chess.PAWN])

    def test_capture_is_not_pawn_push(self):
        """Tests that a capture is not also rewarded as a pawn push."""
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/P7/8/1PPPPPPP/RNBQKBNR w KQkq - 0 2")
        board.push_san("a5")
        board.push_san("b5")
        move = chess.Move.from_uci("a5b6")
        score = self.classifier.score_move(board, move)
        self.assertAlmostEqual(score, self.classifier.piece_values[chess.PAWN])

    def test_king_safety_penalty(self):
        """Tests that moving the king into a heavily attacked area is penalized."""
        # White king on e1 is safe. Black queen on a2 covers d2, e2, f2.
        board = chess.Board("rnb1kbnr/pp2pppp/8/2p5/2p5/8/q1PPPPPP/RNBQKBNR w KQkq - 0 5")
        
        # Move Ke1-f1 (still safe)
        move_safe = chess.Move.from_uci("e1f1")
        score_safe = self.classifier._score_king_safety(board, move_safe)
        self.assertAlmostEqual(score_safe, 0.0)
        
        # Move Ke1-d1 (moves into attacked squares)
        move_unsafe = chess.Move.from_uci("e1d1")
        score_unsafe = self.classifier._score_king_safety(board, move_unsafe)
        self.assertAlmostEqual(score_unsafe, self.classifier.king_safety_penalty)

    def test_zero_score_move(self):
        """Tests that a quiet move gets a score of zero."""
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b KQkq - 1 1")
        move = chess.Move.from_uci("a7a6")
        self.assertAlmostEqual(self.classifier.score_move(board, move), 0.0)

if __name__ == '__main__':
    unittest.main()