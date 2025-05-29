import time # Add this import at the top of your chess_environment.py file
import collections # Should already be implicitly available but good to be aware
import chess
import chess.engine
import re # For parsing 'd' command output

class ChessEnvironment:
    def __init__(self, uci_engine_path: str):
        self.board = chess.Board()
        self.engine = None
        try:
            # popen_uci returns a UciProcess object which SimpleEngine wraps.
            # We'll use SimpleEngine for its convenience but access its .process for 'd' command.
            self.engine = chess.engine.SimpleEngine.popen_uci(uci_engine_path)
            # A quick check to ensure the engine is responsive.
            self.engine.analyse(self.board, chess.engine.Limit(nodes=1)) 
            print(f"UCI engine initialized from {uci_engine_path} and seems responsive.")
        except FileNotFoundError:
            print(f"ERROR: UCI engine not found at {uci_engine_path}. Please check the path.")
            raise
        except chess.engine.EngineError as e:
            print(f"ERROR: Failed to initialize or communicate with UCI engine: {e}")
            if self.engine: # If engine object exists but failed configure/ping
                try:
                    self.engine.quit()
                except chess.engine.EngineError:
                    pass # Ignore errors during quit if init already failed
            raise
        self.reset()

    def reset(self):
        self.board.reset()
        print("DEBUG: reset() called. Board reset. No ucinewgame attempt in this version.")
        if not self.engine:
            print("DEBUG: self.engine is None in reset().")
        elif not hasattr(self.engine, 'protocol'): # Check for 'protocol'
            print("DEBUG: self.engine exists, but no 'protocol' attribute in reset().")
        else:
            # getattr is safer in case protocol is None, though it shouldn't be if engine init succeeded
            print(f"DEBUG: self.engine.protocol is {getattr(self.engine, 'protocol', 'PROTOCOL_IS_NONE')} in reset().")

    def get_current_state_fen(self) -> str: # [cite: 57]
        return self.board.fen()

    def get_legal_moves(self) -> list[str]:
        if not self.engine:
            print("ERROR (Diagnostic): Engine not available for diagnostic test.")
            return []

        print(f"DEBUG (Diagnostic): Current FEN: {self.board.fen()}")
        try:
            engine_name = self.engine.id.get("name", "Unknown")
            engine_author = self.engine.id.get("author", "Unknown")
            print(f"DEBUG (Diagnostic): Engine Name: {engine_name}, Author: {engine_author}")

            if engine_name == "Unknown":
                print("WARNING (Diagnostic): Engine name was not retrieved from initial UCI handshake.")

            print("DEBUG (Diagnostic): Attempting engine.ping()")
            self.engine.ping() 
            print("DEBUG (Diagnostic): engine.ping() successful.")
            
        except chess.engine.EngineTerminatedError:
            print("ERROR (Diagnostic): Engine terminated unexpectedly.")
            self.engine = None 
            raise
        except AttributeError as e: # Specifically to see if readline is the issue here
            print(f"DEBUG (Diagnostic): AttributeError during diagnostic (ping or id access): {e}")
            import traceback
            traceback.print_exc()
            raise 
        except Exception as e:
            print(f"DEBUG (Diagnostic): Other error during diagnostic: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            raise 

        print("DEBUG (Diagnostic): Diagnostic test in get_legal_moves finished. Returning empty list.")
        return []

    def apply_move(self, move_uci: str): # [cite: 59]
        # It's assumed move_uci comes from get_legal_moves() and is thus engine-verified.
        try:
            # `python-chess`'s `board.push_uci()` will validate the UCI string format
            # and the legality of the move according to its own rules.
            self.board.push_uci(move_uci)
            # The engine will be synchronized with the new self.board.fen()
            # when `self.engine.position(self.board)` is called,
            # typically at the start of the next `get_legal_moves()` call.
        except ValueError as e:
            # This can mean malformed UCI or move is illegal by python-chess's standards.
            # If engine provided it, this points to a discrepancy or bug.
            current_fen = self.board.fen()
            print(f"CRITICAL ERROR: Engine-provided move '{move_uci}' rejected by python-chess board.push_uci() for FEN '{current_fen}'. Error: {e}")
            # For debugging, let's see what python-chess thinks are legal moves:
            # print(f"Python-chess internal legal moves: {[m.uci() for m in self.board.legal_moves]}")
            raise chess.engine.EngineError(f"Engine move '{move_uci}' rejected by python-chess board.push_uci() on FEN '{current_fen}'. Discrepancy with engine.")


    def is_game_over(self) -> bool: # [cite: 60]
        return self.board.is_game_over()

    def get_current_player(self) -> bool: # [cite: 61]
        return self.board.turn == chess.WHITE # True if White's turn

    def get_game_outcome(self) -> dict | None: # [cite: 62]
        outcome = self.board.outcome()
        if outcome:
            winner_color = "DRAW"
            if outcome.winner == chess.WHITE:
                winner_color = "WHITE"
            elif outcome.winner == chess.BLACK:
                winner_color = "BLACK"
            
            termination_reason = "UNKNOWN"
            if outcome.termination:
                termination_reason = outcome.termination.name.upper()
            return {"winner": winner_color, "reason": termination_reason}
        return None

    def get_scalar_outcome(self) -> int | None: # [cite: 63]
        outcome = self.board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE: return 1
            if outcome.winner == chess.BLACK: return -1
            return 0 # Draw
        return None # Game not over

    def close_engine(self):
        if self.engine:
            try:
                self.engine.quit()
                print("UCI engine process quit.")
            except chess.engine.EngineError as e:
                print(f"Error quitting engine: {e}")
            except Exception as e:
                print(f"Unexpected error quitting engine: {e}")
            self.engine = None

    def convert_to_gnn_input(self): # [cite: 64]
        raise NotImplementedError("convert_to_gnn_input is not yet implemented.")