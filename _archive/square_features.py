import chess
import dataclasses
from typing import List, Optional

# Define piece types and colors for one-hot encoding
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
COLORS = [chess.WHITE, chess.BLACK]

@dataclasses.dataclass
class SquareFeatures:
    """
    Represents the features for a single square on the chess board.
    """
    # Piece type (one-hot, 7 elements: P, N, B, R, Q, K, Empty)
    piece_type: List[float]
    # Piece color (one-hot, 3 elements: White, Black, None)
    piece_color: List[float]
    # Positional encoding (2 elements: rank_idx, file_idx, 0-7)
    positional_encoding: List[float]
    # Attack/Defense status (4 elements)
    is_attacked_by_white: float
    is_attacked_by_black: float
    is_defended_by_white_piece_on_square: float # If a white piece is on this square, is it defended by white?
    is_defended_by_black_piece_on_square: float # If a black piece is on this square, is it defended by black?
    # Special square status (1 element)
    is_en_passant_target: float

    def to_vector(self) -> List[float]:
        """Converts the features to a single flat vector."""
        return (
            self.piece_type +
            self.piece_color +
            self.positional_encoding +
            [
                self.is_attacked_by_white,
                self.is_attacked_by_black,
                self.is_defended_by_white_piece_on_square,
                self.is_defended_by_black_piece_on_square,
                self.is_en_passant_target,
            ]
        )

    @classmethod
    def get_feature_dimension(cls) -> int:
        """Returns the total dimension of the feature vector."""
        return 7 + 3 + 2 + 4 + 1


def get_square_features(board: chess.Board, square_index: chess.Square) -> SquareFeatures:
    """
    Computes the features for a given square on the board.

    Args:
        board: The python-chess board object.
        square_index: The integer index of the square (0-63, A1=0, H8=63).

    Returns:
        A SquareFeatures object for the given square.
    """
    piece = board.piece_at(square_index)

    # 1. Piece Type
    piece_type_feature = [0.0] * 7
    if piece:
        try:
            idx = PIECE_TYPES.index(piece.piece_type)
            piece_type_feature[idx] = 1.0
        except ValueError:
            pass # Should not happen if PIECE_TYPES is complete
    else:
        piece_type_feature[6] = 1.0 # Empty square

    # 2. Piece Color
    piece_color_feature = [0.0] * 3
    if piece:
        if piece.color == chess.WHITE:
            piece_color_feature[0] = 1.0
        elif piece.color == chess.BLACK:
            piece_color_feature[1] = 1.0
    else:
        piece_color_feature[2] = 1.0 # No piece, so no color

    # 3. Positional Encoding
    rank = chess.square_rank(square_index) # 0-7
    file = chess.square_file(square_index) # 0-7
    positional_encoding = [float(rank), float(file)]

    # 4. Attack/Defense Status
    is_attacked_by_white = 0.0
    if board.is_attacked_by(chess.WHITE, square_index):
        is_attacked_by_white = 1.0

    is_attacked_by_black = 0.0
    if board.is_attacked_by(chess.BLACK, square_index):
        is_attacked_by_black = 1.0

    is_defended_by_white_piece = 0.0
    is_defended_by_black_piece = 0.0
    if piece:
        # Check if the piece on this square is defended by its own color
        # Temporarily remove the piece to check for attackers (which are defenders of this piece)
        original_piece = board.remove_piece_at(square_index)
        if piece.color == chess.WHITE:
            if board.is_attacked_by(chess.WHITE, square_index):
                is_defended_by_white_piece = 1.0
        elif piece.color == chess.BLACK:
            if board.is_attacked_by(chess.BLACK, square_index):
                is_defended_by_black_piece = 1.0
        board.set_piece_at(square_index, original_piece) # Restore the piece


    # 5. Special Square Status
    is_en_passant_target = 0.0
    if board.ep_square == square_index:
        # Check if the en passant capture is actually legal (e.g., pawn in correct position)
        # For simplicity now, just mark the ep_square.
        # A more rigorous check would involve seeing if a pawn of the current player can capture to board.ep_square
        is_en_passant_target = 1.0


    return SquareFeatures(
        piece_type=piece_type_feature,
        piece_color=piece_color_feature,
        positional_encoding=positional_encoding,
        is_attacked_by_white=is_attacked_by_white,
        is_attacked_by_black=is_attacked_by_black,
        is_defended_by_white_piece_on_square=is_defended_by_white_piece,
        is_defended_by_black_piece_on_square=is_defended_by_black_piece,
        is_en_passant_target=is_en_passant_target,
    )