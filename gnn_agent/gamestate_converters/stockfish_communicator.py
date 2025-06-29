import subprocess
import threading
import queue
import time
import os
import io
import chess

class StockfishCommunicator:
    # --- MODIFICATION: Accept an optional Elo rating in the constructor ---
    def __init__(self, stockfish_path: str, elo: int | None = None):
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(
                f"Stockfish executable not found at '{stockfish_path}'. "
                "Please ensure Stockfish is installed and the path is correct."
            )
        if not os.access(stockfish_path, os.X_OK):
            raise PermissionError(
                f"Stockfish executable at '{stockfish_path}' is not executable. "
                "Please check file permissions."
            )

        self.stockfish_path = stockfish_path
        self.elo = elo  # --- Store the Elo rating ---
        self.process = None
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()
        self._stdout_thread = None
        self._stderr_thread = None
        self._running = False
        
        # Internal python-chess board to track game state
        self.board = chess.Board()

    def _enqueue_output(self, pipe, q, pipe_name):
        try:
            for line_str in iter(pipe.readline, ''):
                line_str = line_str.strip()
                if line_str:
                    q.put(line_str)
        except ValueError:
            pass # Ignore errors on pipe close
        except Exception as e:
            pass # Ignore other errors
        finally:
            q.put(None) # Sentinel to indicate pipe has closed

    def start_engine(self) -> bool:
        if self.is_process_alive():
            print("Stockfish engine is already running.")
            return True
        try:
            creationflags = 0
            if os.name == 'nt':
                creationflags = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1,
                creationflags=creationflags
            )
            self._running = True

            self._stdout_thread = threading.Thread(
                target=self._enqueue_output,
                args=(self.process.stdout, self.stdout_queue, "stdout"),
                daemon=True
            )
            self._stderr_thread = threading.Thread(
                target=self._enqueue_output,
                args=(self.process.stderr, self.stderr_queue, "stderr"),
                daemon=True
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            print(f"Stockfish process started (PID: {self.process.pid}).")
            return True
        except (FileNotFoundError, PermissionError, Exception) as e:
            print(f"ERROR: Failed to start Stockfish process: {e}")
            self._running = False
            return False

    def _send_command(self, command: str):
        if self.process and self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.write(f"{command}\n")
                self.process.stdin.flush()
            except (BrokenPipeError, Exception) as e:
                print(f"ERROR sending command '{command}': {e}")
                self._running = False
        else:
            print(f"ERROR: Cannot send command '{command}'. Stockfish process not running or stdin closed.")
            self._running = False

    def _check_stderr(self, print_all=False):
        error_lines = []
        while True:
            try:
                err_line = self.stderr_queue.get_nowait()
                if err_line is None:
                    break
                error_lines.append(err_line)
                if print_all or "error" in err_line.lower():
                    print(f"STDERR: {err_line}")
            except queue.Empty:
                break
        return error_lines

    def _raw_uci_command_exchange(self, command: str, expected_response_token: str, timeout: float = 10.0) -> tuple[bool, list[str]]:
        if not self.is_process_alive():
            print(f"ERROR: Cannot send '{command}'; Stockfish process is not running.")
            return False, []

        self._send_command(command)
        success, lines = self._read_output_until(expected_response_token, timeout)

        if not success:
            print(f"ERROR: Failed to receive '{expected_response_token}' for command '{command}'.")
            if not self.is_process_alive():
                print("ERROR: Stockfish process is not alive after command exchange attempt.")
        return success, lines

    def _read_output_until(self, expected_token: str, timeout: float = 10.0) -> tuple[bool, list[str]]:
        lines_read = []
        start_time = time.time()
        token_found = False

        while time.time() - start_time < timeout:
            if not self.is_process_alive():
                while True:
                    try:
                        line = self.stdout_queue.get_nowait()
                        if line is None:
                            self._running = False
                            break
                        lines_read.append(line)
                        if expected_token in line:
                            token_found = True
                    except queue.Empty:
                        break
                if not token_found:
                    print(f"ERROR: Stockfish process terminated before '{expected_token}' was found.")
                break

            try:
                line = self.stdout_queue.get(timeout=0.05)
                if line is None:
                    self._running = False
                    break
                lines_read.append(line)
                if expected_token in line:
                    token_found = True
                    break
            except queue.Empty:
                continue
        
        if not token_found and (time.time() - start_time >= timeout) and self.is_process_alive():
            print(f"WARNING: Timeout ({timeout}s) waiting for '{expected_token}'. Engine process is still alive.")
        
        self._check_stderr(print_all=not token_found)
        return token_found, lines_read

    def perform_handshake(self) -> bool:
        if not self.is_process_alive():
            if not self.start_engine():
                print("ERROR: Handshake failed - Stockfish engine could not be started.")
                return False
            time.sleep(0.2)

        print("--- Starting UCI Handshake ---")
        print("Sending 'uci' command...")
        uci_success, _ = self._raw_uci_command_exchange("uci", "uciok", timeout=15.0)
        if not uci_success:
            print("ERROR: UCI handshake failed: Did not receive 'uciok'.")
            self.close()
            return False
        print("'uciok' received.")

        # --- MODIFICATION: Configure Elo rating if it was provided ---
        if self.elo is not None:
            print(f"Configuring Stockfish with Elo rating: {self.elo}")
            self._send_command("setoption name UCI_LimitStrength value true")
            self._send_command(f"setoption name UCI_Elo value {self.elo}")

        print("Sending 'isready' command...")
        ready_success, _ = self._raw_uci_command_exchange("isready", "readyok", timeout=10.0)
        if not ready_success:
            print("ERROR: UCI handshake failed: Did not receive 'readyok'.")
            self.close()
            return False
        print("'readyok' received.")
        print("--- Stockfish UCI Handshake Successful ---")
        return True

    def is_process_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def close(self):
        print("--- Closing Stockfish Communicator ---")
        self._running = False
        if self.process:
            if self.process.stdin and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except Exception as e:
                    print(f"WARNING: Error closing Stockfish stdin: {e}")
            
            if self._stdout_thread and self._stdout_thread.is_alive():
                self._stdout_thread.join(timeout=1.0)
            if self._stderr_thread and self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1.0)

            if self.is_process_alive():
                print("Terminating Stockfish process...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                    print("Stockfish process terminated.")
                except subprocess.TimeoutExpired:
                    print("WARNING: Stockfish process did not terminate gracefully, killing.")
                    self.process.kill()
                    self.process.wait()
            self.process = None
        print("--- Stockfish Communicator Closed ---")

    ### --- GAME STATE METHODS --- ###

    def reset_board(self):
        """
        Resets the internal Stockfish game and the python-chess board.
        """
        if not self.is_process_alive():
            print("ERROR: Cannot reset board, Stockfish process is not running.")
            return

        self._send_command("position startpos")
        self.board.reset()

        success, _ = self._raw_uci_command_exchange("isready", "readyok")
        if not success:
            print("ERROR: Engine did not become ready after reset_board command.")

    def make_move(self, uci_move: str):
        """
        Makes a move on the internal board and updates Stockfish's position.
        """
        if not self.is_process_alive():
            print(f"ERROR: Cannot make move {uci_move}, Stockfish not running.")
            return

        try:
            move = chess.Move.from_uci(uci_move)
            if move in self.board.legal_moves:
                self.board.push(move)
                self._send_command(f"position fen {self.board.fen()}")
                success, _ = self._raw_uci_command_exchange("isready", "readyok")
                if not success:
                    print(f"ERROR: Engine not ready after making move {uci_move}.")
            else:
                raise ValueError(f"Illegal move {uci_move} for FEN {self.board.fen()}")
        except ValueError as e:
            print(f"ERROR: {e}")

    def is_game_over(self) -> bool:
        """Checks if the game is over using the internal python-chess board."""
        return self.board.is_game_over(claim_draw=True)

    def get_game_outcome(self) -> float:
        """
        Gets the game outcome from the internal board.
        Returns 1.0 for White win, -1.0 for Black win, 0.0 for Draw.
        """
        outcome = self.board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE:
                return 1.0
            elif outcome.winner == chess.BLACK:
                return -1.0
        return 0.0

    def get_best_move(self, depth: int, timeout: float = 20.0) -> str | None:
        """
        Asks Stockfish to calculate and return the best move from the current board state.
        
        Args:
            depth (int): The search depth for Stockfish to use.
            timeout (float): Max time in seconds to wait for a move.
        
        Returns:
            str | None: The best move in UCI format (e.g., "e2e4") or None if failed.
        """
        if not self.is_process_alive():
            print("ERROR: Cannot get best move, Stockfish is not running.")
            return None

        self._send_command(f"go depth {depth}")
        
        token_found, lines = self._read_output_until("bestmove", timeout=timeout)

        if token_found:
            for line in reversed(lines):
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]

        print(f"ERROR: Stockfish did not return a 'bestmove' within {timeout}s for depth {depth}.")
        return None