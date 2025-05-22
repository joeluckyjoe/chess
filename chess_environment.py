# chess_environment.py
import chess

class ChessEnvironment:
    def __init__(self):
        """
        Initializes a new chess board in the standard starting position.
        """
        self.board = chess.Board()

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