import unittest
import chess
from square_features import get_square_features, SquareFeatures, PIECE_TYPES

class TestSquareFeatures(unittest.TestCase):

    def test_initial_board_e2_square(self):
        board = chess.Board()
        e2_square_index = chess.E2
        features = get_square_features(board, e2_square_index)

        # Piece Type (Pawn)
        expected_piece_type = [0.0] * 7
        expected_piece_type[PIECE_TYPES.index(chess.PAWN)] = 1.0
        self.assertEqual(features.piece_type, expected_piece_type)

        # Piece Color (White)
        expected_piece_color = [1.0, 0.0, 0.0] # White, Black, None
        self.assertEqual(features.piece_color, expected_piece_color)

        # Positional Encoding (E2 -> file 4, rank 1)
        self.assertEqual(features.positional_encoding, [1.0, 4.0]) # rank_idx 1, file_idx 4

        # Attack/Defense (E2 pawn in initial position)
        # E2 is attacked by other white pieces (e.g., D1 Queen, F1 Bishop, G1 Knight, E1 King)
        self.assertEqual(features.is_attacked_by_white, 1.0) # CORRECTED: E2 is attacked by other white pieces
        self.assertEqual(features.is_attacked_by_black, 0.0) # Not attacked by black
        # The E2 pawn is defended by D1 Queen, E1 King, F1 Bishop, G1 Knight
        self.assertEqual(features.is_defended_by_white_piece_on_square, 1.0)
        self.assertEqual(features.is_defended_by_black_piece_on_square, 0.0)

        # Special Square Status
        self.assertEqual(features.is_en_passant_target, 0.0)
        self.assertEqual(len(features.to_vector()), SquareFeatures.get_feature_dimension())

    def test_empty_square_e4_initial_board(self):
        board = chess.Board()
        e4_square_index = chess.E4
        features = get_square_features(board, e4_square_index)

        # Piece Type (Empty)
        expected_piece_type = [0.0] * 6 + [1.0] # Last element is Empty
        self.assertEqual(features.piece_type, expected_piece_type)

        # Piece Color (None)
        expected_piece_color = [0.0, 0.0, 1.0] # White, Black, None
        self.assertEqual(features.piece_color, expected_piece_color)

        # Positional Encoding (E4 -> file 4, rank 3)
        self.assertEqual(features.positional_encoding, [3.0, 4.0])

        # Attack/Defense (E4 in initial board)
        self.assertEqual(features.is_attacked_by_white, 0.0) # No white piece currently attacks e4 directly
        self.assertEqual(features.is_attacked_by_black, 0.0) # No black piece currently attacks e4 directly
        self.assertEqual(features.is_defended_by_white_piece_on_square, 0.0) # No piece on e4
        self.assertEqual(features.is_defended_by_black_piece_on_square, 0.0) # No piece on e4

        self.assertEqual(features.is_en_passant_target, 0.0)
        self.assertEqual(len(features.to_vector()), SquareFeatures.get_feature_dimension())

    def test_en_passant_target_square(self):
        # White pawn on e5, black just moved d7-d5. e.p. target is d6 for white.
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPP2PPP/RNBQKBNR w KQkq d6 0 1")
        d6_square_index = chess.D6 # The en passant target square for white

        e5_features = get_square_features(board, chess.E5)
        self.assertEqual(e5_features.piece_type[PIECE_TYPES.index(chess.PAWN)], 1.0)
        self.assertEqual(e5_features.piece_color, [1.0, 0.0, 0.0]) # White

        d5_features = get_square_features(board, chess.D5)
        self.assertEqual(d5_features.piece_type[PIECE_TYPES.index(chess.PAWN)], 1.0)
        self.assertEqual(d5_features.piece_color, [0.0, 1.0, 0.0]) # Black

        d6_features = get_square_features(board, d6_square_index)
        self.assertEqual(d6_features.is_en_passant_target, 1.0)
        self.assertTrue(board.is_en_passant(chess.Move(chess.E5, chess.D6)))

        c6_features = get_square_features(board, chess.C6)
        self.assertEqual(c6_features.is_en_passant_target, 0.0)

    def test_attack_and_defense_scenario(self):
        # Scenario: White Rook on a1, Black King on c1. White Queen on b2.
        # Square c1 is attacked by white rook (a1) and queen (b2).
        # Black King on c1 is attacked.
        # White Rook on a1 is defended by White Queen on b2.
        # CORRECTED FEN:
        board = chess.Board("8/8/8/8/8/8/1Q6/R1k5 w - - 0 1") # Ra1, kc1, Qb2
        
        c1_idx = chess.C1 # Black King is on c1
        features_c1 = get_square_features(board, c1_idx)

        self.assertEqual(features_c1.piece_type[PIECE_TYPES.index(chess.KING)], 1.0)
        self.assertEqual(features_c1.piece_color, [0.0, 1.0, 0.0]) # Black King
        self.assertEqual(features_c1.is_attacked_by_white, 1.0) # Attacked by Ra1 and Qb2
        self.assertEqual(features_c1.is_attacked_by_black, 0.0) # King doesn't self-attack for this purpose
        self.assertEqual(features_c1.is_defended_by_white_piece_on_square, 0.0) # Black piece here
        self.assertEqual(features_c1.is_defended_by_black_piece_on_square, 0.0) # King has no other black piece defenders

        a1_idx = chess.A1 # White Rook is on a1
        features_a1 = get_square_features(board, a1_idx) 
        self.assertEqual(features_a1.piece_type[PIECE_TYPES.index(chess.ROOK)], 1.0)
        self.assertEqual(features_a1.piece_color, [1.0, 0.0, 0.0]) # White Rook
        self.assertEqual(features_a1.is_defended_by_white_piece_on_square, 1.0) # Defended by Qb2

    def test_feature_vector_conversion(self):
        board = chess.Board()
        features = get_square_features(board, chess.A1)
        vector = features.to_vector()
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), SquareFeatures.get_feature_dimension())
        self.assertTrue(all(isinstance(x, float) for x in vector))

if __name__ == '__main__':
    unittest.main()