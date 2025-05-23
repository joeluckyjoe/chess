# chess_environment.py
import chess

class ChessEnvironment:
    def __init__(self):
        """
        Initializes a new chess board in the standard starting position.
        """
        self.board = chess.Board()

    def reset(self):
        """
        Resets the board to the standard starting position.
        """
        self.board.reset()

    def get_current_state_fen(self):
        """
        Returns the current board state in FEN (Forsyth-Edwards Notation) format.
        FEN is a standard text notation for describing a particular board position.
        """
        return self.board.fen()

    def get_legal_moves(self):
        """
        Returns a list of legal moves from the current position.
        Moves are represented in UCI (Universal Chess Interface) format (e.g., 'e2e4').
        """
        return [move.uci() for move in self.board.legal_moves]

    def get_current_player(self):
        """
        Returns the current player to move.
        True for White, False for Black.
        """
        return self.board.turn == chess.WHITE
    
    def apply_move(self, move_uci: str):
        """
        Applies a move to the board.
        The move must be in UCI format (e.g., 'e2e4').
        Raises an ValueError if the move is illegal or not in correct format.
        """
        try:
            move = self.board.parse_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
            else:
                raise ValueError(f"Illegal move: {move_uci} in current position.")
        except ValueError as e:
            # Catch parsing errors from parse_uci as well (e.g. malformed UCI)
            raise ValueError(f"Invalid or illegal move: {move_uci}. {e}")


    def is_game_over(self):
        """
        Checks if the game is over (checkmate, stalemate, draw).
        """
        return self.board.is_game_over()

    def get_game_outcome(self):
        """
        Returns the outcome of the game if it's over.
        - If White won: {"winner": "WHITE", "reason": "CHECKMATE" or "RESIGNATION" (etc.)}
        - If Black won: {"winner": "BLACK", "reason": "CHECKMATE" or "RESIGNATION" (etc.)}
        - If Draw: {"winner": "DRAW", "reason": "STALEMATE", "FIFTY_MOVE_RULE", etc.}
        - If game not over: None
        """
        if not self.board.is_game_over():
            return None

        result = self.board.result(claim_draw=True) # claim_draw=True considers draw offers

        outcome = {}
        if result == "1-0":
            outcome["winner"] = "WHITE"
        elif result == "0-1":
            outcome["winner"] = "BLACK"
        elif result == "1/2-1/2":
            outcome["winner"] = "DRAW"
        else: # Should not happen with standard results
            return {"winner": "UNKNOWN", "reason": result}

        # Add reason for game termination
        if self.board.is_checkmate():
            outcome["reason"] = "CHECKMATE"
        elif self.board.is_stalemate():
            outcome["reason"] = "STALEMATE"
        elif self.board.is_insufficient_material():
            outcome["reason"] = "INSUFFICIENT_MATERIAL"
        elif self.board.is_seventyfive_moves(): # Or is_fifty_moves if you don't auto-claim
            outcome["reason"] = "SEVENTY_FIVE_MOVE_RULE"
        elif self.board.is_fivefold_repetition():
            outcome["reason"] = "FIVEFOLD_REPETITION"
        # Note: python-chess does not directly tell you if a draw was by agreement or resignation
        # This outcome structure is a common way to represent it.
        # The value head in your NN expects -1, 0, +1[cite: 14], we'll map to that later.
        return outcome
    
    def get_scalar_outcome(self):
        """
        Returns the game outcome as a scalar:
        -  1: White won.
        - -1: Black won.
        -  0: Draw.
        Returns None if the game is not over.
        """
        if not self.board.is_game_over():
            return None

        # The board.result() gives result from White's perspective:
        # "1-0" (White won), "0-1" (Black won), "1/2-1/2" (Draw)
        result_str = self.board.result(claim_draw=True)

        if result_str == "1-0":
            return 1  # White won
        elif result_str == "0-1":
            return -1 # Black won
        elif result_str == "1/2-1/2":
            return 0  # Draw
        else:
            # Should not happen with a completed game using standard rules
            # Or could be for ongoing games if is_game_over() was false
            return None # Or raise an error for unexpected result string