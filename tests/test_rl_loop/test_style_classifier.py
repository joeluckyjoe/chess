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
        # This pawn capture crosses the midline, but should only get the capture reward.
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/P7/8/1PPPPPPP/RNBQKBNR w KQkq - 0 2")
        board.push_san("a5")
        board.push_san("b5")
        move = chess.Move.from_uci("a5b6") # White captures on b6
        score = self.classifier.score_move(board, move)
        self.assertAlmostEqual(score, self.classifier.piece_values[chess.PAWN])

    def test_zero_score_move(self):
        """Tests that a quiet move gets a score of zero."""
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b KQkq - 1 1")
        move = chess.Move.from_uci("a7a6")
        self.assertAlmostEqual(self.classifier.score_move(board, move), 0.0)

if __name__ == '__main__':
    unittest.main()