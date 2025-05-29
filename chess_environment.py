import chess
from stockfish_communicator import StockfishCommunicator

class ChessEnvironmentInterface:
    """
    Manages the chess game state and interactions, using python-chess
    for internal representation and Stockfish (via StockfishCommunicator)
    for authoritative legal move generation. [cite: 3, 24]
    """

    def __init__(self, stockfish_path: str):
        """
        Initializes the chess environment.

        Args:
            stockfish_path (str): The path to the Stockfish executable.
        """
        print("Initializing Chess Environment Interface...")
        self.board = chess.Board() # Internal python-chess board [cite: 83]
        try:
            # Initialize and perform handshake with Stockfish [cite: 83]
            self.communicator = StockfishCommunicator(stockfish_path)
            print("Performing initial Stockfish handshake...")
            if not self.communicator.perform_handshake():
                self.communicator.close() # Attempt to clean up
                raise RuntimeError("Failed to establish handshake with Stockfish engine.")
            print("Stockfish handshake successful.")
        except Exception as e:
            print(f"FATAL ERROR during StockfishCommunicator initialization: {e}")
            raise RuntimeError(f"Could not initialize StockfishCommunicator: {e}")
        
        print("Chess Environment Interface initialized.")

    def reset(self) -> str:
        """
        Resets the environment to the standard starting position. [cite: 86]

        Returns:
            str: The FEN string of the starting position.
        """
        print("Resetting Chess Environment...")
        self.board.reset()
        # No need to reset Stockfish if we send FEN each time,
        # but we could send "position startpos" if we wanted.
        return self.get_current_fen()

    def get_current_fen(self) -> str:
        """
        Returns the FEN string of the current board position. [cite: 88]
        """
        return self.board.fen()

    def get_legal_moves(self) -> list[str]:
        """
        Gets the authoritative list of legal moves in UCI format from the
        Stockfish engine for the current board state. (Implements Step 2.4) [cite: 89]

        Returns:
            list[str]: A list of legal moves.
        
        Raises:
            RuntimeError: If communication with the engine fails.
        """
        current_fen = self.get_current_fen()
        # print(f"DEBUG (get_legal_moves): Fetching moves for FEN: {current_fen}")
        moves = self.communicator.get_legal_moves_for_fen(current_fen)
        
        if moves is None:
            # Communicator failed to get moves. This is a critical error.
            raise RuntimeError(f"Failed to get legal moves from Stockfish for FEN: {current_fen}")
            
        # Optional: Compare with python-chess moves for debugging?
        # internal_moves = [m.uci() for m in self.board.legal_moves]
        # if set(moves) != set(internal_moves):
        #     print(f"WARNING: Stockfish moves ({len(moves)}) differ from python-chess ({len(internal_moves)})")
        #     print(f"  Stockfish - python-chess: {set(moves) - set(internal_moves)}")
        #     print(f"  python-chess - Stockfish: {set(internal_moves) - set(moves)}")

        return moves

    def apply_move(self, move_uci: str):
        """
        Applies a UCI move to the internal python-chess board. [cite: 90]
        It's *strongly recommended* that the move_uci comes from the
        get_legal_moves() list to ensure consistency.

        Args:
            move_uci (str): The move to apply in UCI format.

        Raises:
            ValueError: If the move is illegal or malformed according to python-chess.
                        This *should not* happen if the move came from get_legal_moves().
        """
        try:
            move = chess.Move.from_uci(move_uci)
            self.board.push(move) # [cite: 90, 92]
            # print(f"Applied move {move_uci}. New FEN: {self.board.fen()}")
        except ValueError as e:
            print(f"ERROR applying move '{move_uci}' to FEN '{self.board.fen()}'.")
            print(f"  Internal python-chess error: {e}")
            # Consider raising or handling this more gracefully.
            # If this happens, it implies a mismatch between Stockfish and python-chess
            # or an invalid move was somehow selected.
            raise

    def is_game_over(self) -> bool:
        """
        Checks if the game has ended (checkmate, stalemate, draw, etc.). [cite: 93]
        """
        return self.board.is_game_over()

    def get_game_outcome(self) -> chess.Outcome | None:
        """
        Returns the game outcome if it's over, otherwise None. [cite: 94]
        """
        return self.board.outcome()

    def get_scalar_outcome(self) -> int | None:
        """
        Returns the scalar outcome (1 for White win, -1 for Black win, 0 for draw). [cite: 95]
        Returns None if the game is not over.
        """
        outcome = self.board.outcome()
        if outcome is None:
            return None
        if outcome.winner == chess.WHITE:
            return 1
        if outcome.winner == chess.BLACK:
            return -1
        return 0 # Draw

    def get_python_chess_board(self) -> chess.Board:
        """
        Returns the internal python-chess Board object. [cite: 95]
        """
        return self.board

    def close(self):
        """
        Closes the communication with the Stockfish engine.
        """
        print("Closing Chess Environment Interface...")
        if self.communicator:
            self.communicator.close()
        print("Chess Environment Interface closed.")