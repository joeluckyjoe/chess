# gnn_agent/gamestate_converters/action_space_converter.py

import chess

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    Converts a chess.Move object to its corresponding index in the action space.
    
    NOTE: This is a placeholder implementation for testing. A full, robust
    implementation that correctly maps every possible move is required.
    For now, we'll use a simplified mapping that works for the start position.
    """
    # A real implementation would be complex. For now, we'll just use a
    # simplified hash that is unique enough for initial testing.
    # This is NOT a correct or complete mapping.
    from_sq = move.from_square
    to_sq = move.to_square
    return (from_sq * 64) + to_sq