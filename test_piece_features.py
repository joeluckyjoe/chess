import unittest
import chess
from piece_features import PieceFeatures, get_piece_features, PIECE_TYPE_TO_INDEX

class TestPieceFeatures(unittest.TestCase):

    def test_white_pawn_e2_start_pos(self):
        board = chess.Board() # Standard starting position
        e2_square = chess.E2

        features = get_piece_features(board, e2_square)
        self.assertIsNotNone(features)

        # Expected piece type: PAWN (index 0)
        expected_piece_type = [0] * 6
        expected_piece_type[PIECE_TYPE_TO_INDEX[chess.PAWN]] = 1
        self.assertEqual(features.piece_type, expected_piece_type, "Piece Type mismatch")

        # Expected piece color: WHITE (index 0)
        expected_piece_color = [0] * 2
        expected_piece_color[0] = 1 # White
        self.assertEqual(features.piece_color, expected_piece_color, "Piece Color mismatch")

        # Expected location: e2 (rank 1, file 4)
        self.assertEqual(features.rank_idx, 1, "Rank index mismatch") # Rank 2 is index 1
        self.assertEqual(features.file_idx, 4, "File index mismatch") # File 'e' is index 4

        # Expected mobility: 2 (e3, e4)
        self.assertEqual(features.mobility, 2, "Mobility mismatch")

        # Expected attacks_target_count: 0 (d3 and f3 are empty)
        self.assertEqual(features.attacks_target_count, 0, "Attacks target count mismatch")

        # Expected defends_target_count: 0 (Pawn on e2 attacks d3 and f3, which are empty)
        self.assertEqual(features.defends_target_count, 0, "Defends target count mismatch")

        # Expected is_attacked_by_opponent_count: 0
        self.assertEqual(features.is_attacked_by_opponent_count, 0, "Is attacked by opponent count mismatch")

        # Expected is_defended_by_own_count: 4 (Qd1, Ke1, Bf1, Bc1 attack e2)
        self.assertEqual(features.is_defended_by_own_count, 4, "Is defended by own count mismatch")

        # Expected is_pinned: 0
        self.assertEqual(features.is_pinned, 0, "Is pinned mismatch")

        # Expected is_checking: 0
        self.assertEqual(features.is_checking, 0, "Is checking mismatch")

        feature_list = features.to_list()
        self.assertEqual(len(feature_list), features.feature_dimension)
        self.assertEqual(features.feature_dimension, 17)

    def test_white_rook_checking_complex_scenario(self):
        # FEN: Black King on d8, Black Pawn on d7, Black Knight on f6.
        # White King on a1, White Rook on d5, White Bishop on g5, White Pawn on e4.
        # White's turn. White Rook on d5 is checking the Black King on d8.
        fen = "3k4/3p4/5n2/3R2B1/4P3/8/8/K7 w - - 0 1"
        board = chess.Board(fen)
        rd5_square = chess.D5 # White Rook

        features = get_piece_features(board, rd5_square)
        self.assertIsNotNone(features)

        # Piece Type: ROOK
        expected_piece_type = [0] * 6
        expected_piece_type[PIECE_TYPE_TO_INDEX[chess.ROOK]] = 1
        self.assertEqual(features.piece_type, expected_piece_type, "Rook: Piece Type")

        # Piece Color: WHITE
        expected_piece_color = [0] * 2
        expected_piece_color[0] = 1
        self.assertEqual(features.piece_color, expected_piece_color, "Rook: Piece Color")

        # Location: d5 (rank 4, file 3)
        self.assertEqual(features.rank_idx, 4, "Rook: Rank index") # Rank 5 is index 4
        self.assertEqual(features.file_idx, 3, "Rook: File index") # File 'd' is index 3

        # Mobility: Rd5 can move to d6, d7(xP), c5,b5,a5, e5, d4,d3,d2,d1 = 10 moves
        # All these moves result in check or checkmate.
        self.assertEqual(features.mobility, 11, "Rook: Mobility")

        # attacks_target_count: Attacks Black Pawn on d7 and Black King on d8 = 2
        self.assertEqual(features.attacks_target_count, 1, "Rook: Attacks target count")

        # defends_target_count: Attacks (defends) White Bishop on g5 = 1
        self.assertEqual(features.defends_target_count, 1, "Rook: Defends target count")

        # is_attacked_by_opponent_count: Attacked by Black Knight on f6 = 1
        self.assertEqual(features.is_attacked_by_opponent_count, 1, "Rook: Is attacked by opponent")

        # is_defended_by_own_count: Attacked (defended) by White Pawn on e4 = 1
        self.assertEqual(features.is_defended_by_own_count, 1, "Rook: Is defended by own")

        # is_pinned: No
        self.assertEqual(features.is_pinned, 0, "Rook: Is pinned")

        # is_checking: Yes, Rd5 attacks Black King on d8
        self.assertEqual(features.is_checking, 0, "Rook: Is checking")

    def test_white_pawn_pinned(self):
        # FEN: Black King a8, Black Queen d5. White Pawn d2, White King d1.
        # White Pawn on d2 is pinned by Black Queen on d5 against White King on d1.
        fen = "k7/8/8/3q4/8/8/3P4/3K4 w - - 0 1"
        board = chess.Board(fen)
        pd2_square = chess.D2 # White Pawn

        features = get_piece_features(board, pd2_square)
        self.assertIsNotNone(features)

        # Piece Type: PAWN
        expected_piece_type = [0] * 6
        expected_piece_type[PIECE_TYPE_TO_INDEX[chess.PAWN]] = 1
        self.assertEqual(features.piece_type, expected_piece_type, "Pinned Pawn: Piece Type")

        # Piece Color: WHITE
        expected_piece_color = [0] * 2
        expected_piece_color[0] = 1
        self.assertEqual(features.piece_color, expected_piece_color, "Pinned Pawn: Piece Color")

        # Location: d2 (rank 1, file 3)
        self.assertEqual(features.rank_idx, 1, "Pinned Pawn: Rank index") # Rank 2 is index 1
        self.assertEqual(features.file_idx, 3, "Pinned Pawn: File index") # File 'd' is index 3

        # Mobility: Pinned pawn on d2 has 0 moves.
        # (Cannot move to d3 or d4 as it would expose King on d1 to Queen on d5)
        self.assertEqual(features.mobility, 2, "Pinned Pawn: Mobility")

        # attacks_target_count: Attacks c3, e3. Both empty. = 0
        self.assertEqual(features.attacks_target_count, 0, "Pinned Pawn: Attacks target count")

        # defends_target_count: Attacks c3, e3. Both empty. = 0
        self.assertEqual(features.defends_target_count, 0, "Pinned Pawn: Defends target count")

        # is_attacked_by_opponent_count: Attacked by Black Queen on d5 = 1
        self.assertEqual(features.is_attacked_by_opponent_count, 1, "Pinned Pawn: Is attacked by opponent")

        # is_defended_by_own_count: Attacked (defended) by White King on d1 = 1
        self.assertEqual(features.is_defended_by_own_count, 1, "Pinned Pawn: Is defended by own")

        # is_pinned: Yes
        self.assertEqual(features.is_pinned, 1, "Pinned Pawn: Is pinned")

        # is_checking: No
        self.assertEqual(features.is_checking, 0, "Pinned Pawn: Is checking")


if __name__ == '__main__':
    unittest.main()