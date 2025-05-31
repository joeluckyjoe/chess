import chess
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PieceFeatures:
    """
    Represents features for a single chess piece for the GNN_pc.
    """
    # Piece Type (one-hot, 6 elements: P, N, B, R, Q, K - no "Empty" here)
    # Order: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
    piece_type: List[int] = field(default_factory=lambda: [0] * 6)

    # Piece Color (one-hot, 2 elements: White, Black - no "None")
    # Order: WHITE, BLACK
    piece_color: List[int] = field(default_factory=lambda: [0] * 2)

    # Location (2 elements: rank_idx, file_idx, 0-7)
    rank_idx: int = 0
    file_idx: int = 0

    # Mobility (1 element: count of legal moves for this piece)
    mobility: int = 0

    # Attack/Defense Counts for this specific piece
    attacks_target_count: int = 0  # How many opponent pieces this piece is attacking
    defends_target_count: int = 0  # How many friendly pieces this piece is defending
    is_attacked_by_opponent_count: int = 0  # How many opponent pieces are attacking this piece
    is_defended_by_own_count: int = 0  # How many friendly pieces are defending this piece

    is_pinned: int = 0  # Is this piece pinned? (1 if pinned, 0 otherwise)
    is_checking: int = 0 # Does this piece deliver a check? (1 if delivering check, 0 otherwise)


    def to_list(self) -> List[float]: # Changed to float for consistency with NN inputs
        """Converts the feature dataclass to a flat list."""
        return (
            [float(x) for x in self.piece_type]
            + [float(x) for x in self.piece_color]
            + [float(self.rank_idx), float(self.file_idx)]
            + [float(self.mobility)]
            + [float(self.attacks_target_count)]
            + [float(self.defends_target_count)]
            + [float(self.is_attacked_by_opponent_count)]
            + [float(self.is_defended_by_own_count)]
            + [float(self.is_pinned)]
            + [float(self.is_checking)]
        )

    @property
    def feature_dimension(self) -> int:
        """Returns the total dimension of the feature vector."""
        return len(self.to_list())

PIECE_TYPE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Original clean version of get_piece_features to paste back after debugging
def get_piece_features(board: chess.Board, square: chess.Square) -> PieceFeatures:
    """
    Extracts features for the piece at the given square on the board.
    Returns None if there is no piece on the square.
    """
    piece = board.piece_at(square)
    if piece is None:
        return None

    features = PieceFeatures()

    # 1. Piece Type
    features.piece_type[PIECE_TYPE_TO_INDEX[piece.piece_type]] = 1

    # 2. Piece Color
    features.piece_color[0 if piece.color == chess.WHITE else 1] = 1

    # 3. Location
    features.rank_idx = chess.square_rank(square)
    features.file_idx = chess.square_file(square)

    # 4. Mobility
    features.mobility = 0
    for move in board.legal_moves:
        if move.from_square == square:
            features.mobility += 1

    # 5. Attack/Defense Counts
    # features.attacks_target_count and features.defends_target_count are already 0 from PieceFeatures init
    attacked_squares_by_this_piece = board.attacks(square)
    for attacked_sq in attacked_squares_by_this_piece:
        target_piece_on_attacked_sq = board.piece_at(attacked_sq) 
        if target_piece_on_attacked_sq:
            if target_piece_on_attacked_sq.color != piece.color:
                features.attacks_target_count += 1
            else:
                features.defends_target_count += 1
    
    # How many opponent pieces are attacking this piece
    features.is_attacked_by_opponent_count = 0
    for attacker_square in board.attackers(not piece.color, square):
        features.is_attacked_by_opponent_count += 1

    # How many friendly pieces are defending this piece
    features.is_defended_by_own_count = 0
    for defender_square in board.attackers(piece.color, square):
        if defender_square != square:
            features.is_defended_by_own_count += 1

    # 6. Is Pinned?
    features.is_pinned = 1 if board.is_pinned(piece.color, square) else 0

    # 7. Is Checking?
    features.is_checking = 0
    opponent_king_square = board.king(not piece.color)
    if opponent_king_square is not None: 
        if square in board.attackers(piece.color, opponent_king_square):
            features.is_checking = 1
            
    return features