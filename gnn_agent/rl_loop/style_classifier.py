import chess
from typing import Dict

class StyleClassifier:
    """
    A simplified placeholder classifier that rewards captures and aggressive pawn pushes,
    and now penalizes poor king safety.
    """

    def __init__(self):
        self.piece_values: Dict[int, int] = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        self.pawn_push_reward: float = 0.1
        # --- NEW: Added a penalty for poor king safety ---
        self.king_safety_penalty: float = -0.5

    def score_move(self, board: chess.Board, move: chess.Move) -> float:
        """Calculates the total style score for a given move."""
        if not board.is_legal(move):
            return 0.0

        score = 0.0
        score += self._score_capture(board, move)
        score += self._score_pawn_push(board, move)
        # --- NEW: Add the king safety score ---
        score += self._score_king_safety(board, move)
        return score

    def _get_attacked_squares_around_king(self, board: chess.Board, king_square: int, attacker_color: bool) -> int:
        """Counts how many squares in a 3x3 grid around the king are attacked."""
        count = 0
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue
                
                checked_rank = chess.square_rank(king_square) + rank_offset
                checked_file = chess.square_file(king_square) + file_offset

                if 0 <= checked_rank < 8 and 0 <= checked_file < 8:
                    square = chess.square(checked_file, checked_rank)
                    if board.is_attacked_by(attacker_color, square):
                        count += 1
        return count

    def _score_king_safety(self, board: chess.Board, move: chess.Move) -> float:
        """
        Calculates a penalty if a move increases the number of squares
        attacked by the opponent around the king.
        """
        turn = board.turn
        king_square_before = board.king(turn)
        
        # This can happen in rare illegal positions, so we safe-guard it.
        if king_square_before is None:
            return 0.0
            
        attacks_before = self._get_attacked_squares_around_king(board, king_square_before, not turn)

        board_after_move = board.copy()
        board_after_move.push(move)
        
        king_square_after = board_after_move.king(turn)
        if king_square_after is None:
            return 0.0

        attacks_after = self._get_attacked_squares_around_king(board_after_move, king_square_after, not turn)

        if attacks_after > attacks_before:
            return self.king_safety_penalty
            
        return 0.0

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