import chess
from typing import Dict

class StyleClassifier:
    """
    A simplified placeholder classifier that rewards captures and aggressive pawn pushes.
    This provides a basic dense reward signal for the Phase BR reinforcement learning loop.
    """

    def __init__(self):
        self.piece_values: Dict[int, int] = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        self.pawn_push_reward: float = 0.1

    def score_move(self, board: chess.Board, move: chess.Move) -> float:
        """Calculates the total style score for a given move."""
        if not board.is_legal(move):
            return 0.0

        score = 0.0
        score += self._score_capture(board, move)
        score += self._score_pawn_push(board, move)
        return score

    def _score_pawn_push(self, board: chess.Board, move: chess.Move) -> float:
        """Rewards non-capture pawn moves across the midline."""
        if board.is_capture(move): return 0.0
        
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE and chess.square_rank(move.from_square) < 4 and chess.square_rank(move.to_square) >= 4:
                return self.pawn_push_reward
            if piece.color == chess.BLACK and chess.square_rank(move.from_square) > 3 and chess.square_rank(move.to_square) <= 3:
                return self.pawn_push_reward
        return 0.0

    def _score_capture(self, board: chess.Board, move: chess.Move) -> float:
        """Rewards captures based on the value of the captured piece."""
        if board.is_en_passant(move):
            return self.piece_values.get(chess.PAWN, 0)
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                return self.piece_values.get(captured_piece.piece_type, 0)
        return 0.0