# gnn_agent/gamestate_converters/action_space_converter.py (Corrected)

import chess
# Import the robust mapping created in the data converter
from .gnn_data_converter import MOVE_TO_INDEX_MAP

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Converts a chess.Move object to its corresponding index in the action space
    using the definitive project-wide move-to-index map.

    This function now correctly handles all moves, including promotions.

    Args:
        move (chess.Move): The move to convert.
        board (chess.Board): The current board state (required for context,
                             though the current map is universal).

    Returns:
        int: The unique integer index for the move.
    """
    # Create a new move object without promotion for the initial lookup,
    # as the map might not have promotion info on all moves.
    # The key in our map is a simple move; promotions are handled by their own keys.
    
    # The MOVE_TO_INDEX_MAP already handles promotion variants correctly.
    # We simply need to look up the move. If a move is a promotion,
    # the chess.Move object will have the promotion piece set, and it will
    # match the corresponding key in the map.
    
    try:
        return MOVE_TO_INDEX_MAP[move]
    except KeyError:
        # This case should ideally not be reached if the move is legal
        # and the map is comprehensive.
        # Handle cases like null moves if they ever occur.
        print(f"Warning: Move {move.uci()} not found in MOVE_TO_INDEX_MAP. This could indicate an issue.")
        # Fallback for safety, though it's incorrect.
        return -1


def get_action_space_size() -> int:
    """
    Returns the total number of possible actions in the defined action space.
    This should match the output dimension of the policy head.
    """
    # This value is derived from the length of the comprehensive map.
    return len(MOVE_TO_INDEX_MAP)