#
# File: gnn_agent/gamestate_converters/action_space_converter.py (Canonical Version)
#
"""
Defines the canonical mapping from chess moves to a unique integer index for the project.
This action space is designed to be comprehensive and fit within the 4672-move limit.

ACTION SPACE STRUCTURE:
- Indices 0-4095: Standard moves (from_square * 64 + to_square)
- Indices 4096-4671: Promotion moves
"""
import chess

ACTION_SPACE_SIZE = 4672
STANDARD_MOVES_LIMIT = 4096 # 64 * 64

# --- Promotion Move Mapping ---
# We have 576 slots for promotions (4672 - 4096).
# A promotion is defined by the 'from' file, the 'to' file, and the promotion piece.
# There are 8 'from' files, 3 'to' files (straight, capture left, capture right).
# For White (rank 6 to 7) and Black (rank 1 to 0).
# 3 directions * 8 files * 4 promotion pieces = 96 combinations per color.
# Total promotions = 192, which fits comfortably within our 576 slots.

PROMOTION_PIECES_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
PROMOTION_PIECE_OFFSET = {piece: i for i, piece in enumerate(PROMOTION_PIECES_ORDER)}

def _get_promotion_index(move: chess.Move) -> int:
    """Calculates a unique index for a promotion move."""
    # This will be called only if move.promotion is not None.
    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)

    # Determine direction: -1 for left capture, 0 for straight, 1 for right capture
    direction = to_file - from_file
    direction_offset = direction + 1 # Maps [-1, 0, 1] to [0, 1, 2]

    # Get offsets
    piece_offset = PROMOTION_PIECE_OFFSET.get(move.promotion, 0)
    
    # Calculate index based on the pawn's starting file and the move characteristics
    # 3 directions * 4 promotion pieces = 12 variations per starting file
    promo_type_index = (direction_offset * len(PROMOTION_PIECES_ORDER)) + piece_offset
    
    # Each of the 8 files has 12 unique promotion types
    file_base_index = from_file * (3 * len(PROMOTION_PIECES_ORDER))
    
    final_promo_index = file_base_index + promo_type_index
    
    # Add base offset to place it in the promotion block of the action space
    return STANDARD_MOVES_LIMIT + final_promo_index


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Converts a chess.Move object to its corresponding canonical index in the action space.
    """
    if move.promotion:
        return _get_promotion_index(move)
    else:
        # Standard move encoding
        return move.from_square * 64 + move.to_square

def get_action_space_size() -> int:
    """
    Returns the total number of possible actions in the defined action space.
    """
    return ACTION_SPACE_SIZE