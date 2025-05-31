import subprocess
import threading
import queue
import time
import os
import io # For io.DEFAULT_BUFFER_SIZE if needed, though bufsize=1 with text=True should be fine

class StockfishCommunicator:
    def __init__(self, stockfish_path: str):
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
        self.process = None
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()
        self._stdout_thread = None
        self._stderr_thread = None
        self._running = False

    def _enqueue_output(self, pipe, q, pipe_name):
        try:
            # When text=True, pipe is a text wrapper.
            # iter(pipe.readline, '') will read until EOF.
            for line_str in iter(pipe.readline, ''): # pipe.readline() now returns str
                line_str = line_str.strip()
                if line_str: # Add to queue only if not empty after strip
                    # print(f"DEBUG QUEUE({pipe_name}): Enqueuing '{line_str}'") # Verbose
                    q.put(line_str)
            # Normal EOF or pipe closed by process exit
        except ValueError:
            # print(f"Info: {pipe_name} pipe likely closed (ValueError).")
            pass
        except Exception as e:
            # print(f"Error in _enqueue_output for {pipe_name}: {e}")
            pass
        finally:
            # print(f"DEBUG: {pipe_name} thread putting sentinel and exiting.")
            q.put(None) # Sentinel value to indicate the stream has ended

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
                text=True,              # Use text mode
                encoding='utf-8',       # Specify encoding
                bufsize=1,              # Line-buffered in text mode
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
        except FileNotFoundError:
            print(f"ERROR: Stockfish executable not found at '{self.stockfish_path}'.")
            self._running = False
            return False
        except PermissionError:
            print(f"ERROR: No permission to execute Stockfish at '{self.stockfish_path}'.")
            self._running = False
            return False
        except Exception as e:
            print(f"ERROR: Failed to start Stockfish process: {e}")
            self._running = False
            return False

    def _send_command(self, command: str):
        if self.process and self.process.stdin and not self.process.stdin.closed:
            try:
                # print(f"DEBUG: Sending command: {command}")
                self.process.stdin.write(f"{command}\n") # Write string directly
                self.process.stdin.flush()
            except BrokenPipeError:
                print("ERROR: Broken pipe. Stockfish process might have terminated unexpectedly.")
                self._running = False
            except Exception as e:
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
                if err_line is None: # Sentinel
                    break
                error_lines.append(err_line)
                if print_all or "error" in err_line.lower():
                     print(f"STDERR: {err_line}")
            except queue.Empty:
                break
        return error_lines

    def _raw_uci_command_exchange(self, command: str, expected_response_token: str, timeout: float = 10.0) -> tuple[bool, list[str]]:
        if not self.is_process_alive(): # Check actual process status
            print(f"ERROR: Cannot send '{command}'; Stockfish process is not running.")
            return False, []

        # print(f"INFO: Sending command: '{command}'")
        self._send_command(command)
        # print(f"INFO: Waiting for '{expected_response_token}'...")
        success, lines = self._read_output_until(expected_response_token, timeout)

        if success:
            # print(f"INFO: Successfully received '{expected_response_token}' for command '{command}'.")
            pass
        else:
            print(f"ERROR: Failed to receive '{expected_response_token}' for command '{command}'.")
            if not self.is_process_alive():
                 print("ERROR: Stockfish process is not alive after command exchange attempt.")
        return success, lines

    def _read_output_until(self, expected_token: str, timeout: float = 10.0) -> tuple[bool, list[str]]:
        lines_read = []
        start_time = time.time()
        token_found = False
        # print(f"DEBUG: _read_output_until looking for '{expected_token}' with timeout {timeout}s")

        while time.time() - start_time < timeout:
            if not self.is_process_alive():
                # print(f"DEBUG: Process not alive. Draining queue for '{expected_token}'.")
                while True:
                    try:
                        line = self.stdout_queue.get_nowait()
                        if line is None:
                            self._running = False
                            break
                        lines_read.append(line)
                        # print(f"DEBUG Engine Output (Post-Mortem): {line}")
                        if expected_token in line:
                            token_found = True
                    except queue.Empty:
                        break
                if not token_found: # Log only if token still not found after draining
                    print(f"ERROR: Stockfish process terminated before '{expected_token}' was found.")
                break # Exit main while loop

            try:
                # Poll queue with a short timeout to remain responsive
                line = self.stdout_queue.get(timeout=0.05) # 50ms poll
                
                if line is None: # Sentinel from _enqueue_output means stream truly ended
                    self._running = False # Signal our reader threads are done
                    # print(f"DEBUG: stdout stream ended (sentinel received) while waiting for '{expected_token}'.")
                    break

                # print(f"DEBUG Engine Output: {line}") # UNCOMMENT FOR VERY VERBOSE OUTPUT
                lines_read.append(line)
                if expected_token in line:
                    token_found = True
                    break
            except queue.Empty:
                # This is normal if the engine is busy or there's a small delay.
                # Loop again to check main timeout and process status.
                continue
        
        if not token_found and (time.time() - start_time >= timeout) and self.is_process_alive():
            print(f"WARNING: Timeout ({timeout}s) waiting for '{expected_token}'. Engine process is still alive.")
        
        self._check_stderr(print_all=not token_found) # Print all stderr if token not found or timeout
        return token_found, lines_read

    def perform_handshake(self) -> bool:
        if not self.is_process_alive(): # Check before attempting to start
            if not self.start_engine():
                print("ERROR: Handshake failed - Stockfish engine could not be started.")
                return False
            time.sleep(0.2) # Give a brief moment for engine to fully initialize pipes

        print("--- Starting UCI Handshake ---")
        print("Sending 'uci' command...")
        uci_success, uci_lines = self._raw_uci_command_exchange("uci", "uciok", timeout=15.0) # Increased timeout slightly
        
        if not uci_success:
            print("ERROR: UCI handshake failed: Did not receive 'uciok'.")
            print("DEBUG: All lines received during 'uci' command attempt:")
            for line_idx, line_content in enumerate(uci_lines):
                print(f"  [{line_idx+1}] > {line_content}")
            self.close()
            return False
        print("'uciok' received.")

        print("Sending 'isready' command...")
        ready_success, ready_lines = self._raw_uci_command_exchange("isready", "readyok", timeout=10.0)
        if not ready_success:
            print("ERROR: UCI handshake failed: Did not receive 'readyok'.")
            print("DEBUG: All lines received during 'isready' command attempt:")
            for line_idx, line_content in enumerate(ready_lines):
                print(f"  [{line_idx+1}] > {line_content}")
            self.close()
            return False
        print("'readyok' received.")

        print("--- Stockfish UCI Handshake Successful ---")
        return True

    def is_process_alive(self) -> bool:
        """Checks if the Stockfish subprocess itself is currently running."""
        return self.process is not None and self.process.poll() is None

    def close(self):
        print("--- Closing Stockfish Communicator ---")
        self._running = False # Signal threads to stop processing new items

        if self.process:
            if self.process.stdin and not self.process.stdin.closed:
                try:
                    # print("DEBUG: Closing stdin.")
                    self.process.stdin.close()
                except Exception as e:
                    print(f"WARNING: Error closing Stockfish stdin: {e}")
            
            # Threads should see _running = False and their iter(pipe.readline, '') will end if pipe closes.
            # Or they will see _running = False and exit after emptying queue.
            # Give threads a moment to process the sentinel or pipe closure.
            if self._stdout_thread and self._stdout_thread.is_alive():
                # print("DEBUG: Waiting for stdout_thread to join...")
                self._stdout_thread.join(timeout=1.0)
            if self._stderr_thread and self._stderr_thread.is_alive():
                # print("DEBUG: Waiting for stderr_thread to join...")
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
                    try:
                        self.process.wait(timeout=2.0)
                        print("Stockfish process killed.")
                    except subprocess.TimeoutExpired:
                        print("ERROR: Stockfish process could not be killed.")
                except Exception as e:
                    print(f"ERROR during Stockfish process termination: {e}")
            else:
                print("Stockfish process was already terminated or not started properly.")
        
        # Drain queues in case threads didn't fully, or if process died abruptly
        # print("DEBUG: Draining queues post-close...")
        # while not self.stdout_queue.empty(): self.stdout_queue.get_nowait()
        # while not self.stderr_queue.empty(): self.stderr_queue.get_nowait()

        self.process = None
        print("--- Stockfish Communicator Closed ---")

    def get_legal_moves_output(self, fen: str) -> list[str] | None:
        """
        Sets the position on the Stockfish engine using FEN and then fetches
        the raw output of 'go perft 1'. The output of 'go perft 1' lists
        legal moves at depth 1, each with a count, and ends with a 'Nodes searched' line.
        This method fulfills Step 2.2 of the plan.

        Args:
            fen (str): The FEN string of the position.
                       Can also be "startpos" to use the standard chess starting position.

        Returns:
            list[str] | None: A list of all output lines from the 'go perft 1' command
                              if successful, None otherwise. These lines will be parsed in Step 2.3.
        """
        if not self.is_process_alive():
            print("ERROR: Stockfish process not running. Cannot get legal moves output.")
            return None

        position_command = f"position fen {fen}"
        if fen.lower() == "startpos":
            position_command = "position startpos"

        # print(f"INFO: Sending position command: '{position_command}'")
        self._send_command(position_command)
        # Note: The "position" command itself doesn't send a direct "ok" acknowledgement.
        # We must ensure the engine has processed it before sending further commands.

        # print("INFO: Ensuring engine is ready after setting position...")
        # Use _raw_uci_command_exchange to send 'isready' and wait for 'readyok'
        isready_success, isready_lines = self._raw_uci_command_exchange("isready", "readyok", timeout=5.0)
        if not isready_success:
            print(f"ERROR: Engine not ready after '{position_command}'. Lines received during 'isready':")
            for line_idx, line_content in enumerate(isready_lines):
                print(f"  [{line_idx+1}] > {line_content}")
            return None
        # print("INFO: Engine is ready.")

        # Now, send 'go perft 1' to get the list of moves and their counts at depth 1.
        # The output will end with a line containing "Nodes searched:".
        # print("INFO: Sending 'go perft 1' to fetch legal moves output...")
        go_success, perft_lines = self._raw_uci_command_exchange("go perft 1", "Nodes searched:", timeout=10.0)

        if not go_success:
            print("ERROR: Failed to get complete output from 'go perft 1'.")
            if perft_lines: # Print what was received, if anything
                print("DEBUG: Lines received during 'go perft 1' attempt:")
                for line_idx, line_content in enumerate(perft_lines):
                    print(f"  [{line_idx+1}] > {line_content}")
            return None

        # print("INFO: Successfully received output from 'go perft 1'.")
        return perft_lines

    # Insert this private helper method for parsing
    def _parse_perft_output_for_uci_moves(self, perft_lines: list[str]) -> list[str]:
        """
        Parses the raw output lines from a 'go perft 1' command to extract
        a list of legal moves in UCI format. (Implements parsing logic for Step 2.3)

        Args:
            perft_lines (list[str]): A list of strings, where each string is a line
                                     from the 'go perft 1' output.

        Returns:
            list[str]: A list of legal moves in UCI format (e.g., ["e2e4", "d2d4"]).
                       Returns an empty list if no moves can be parsed or input is None/empty.
        """
        uci_moves = []
        if not perft_lines:
            return uci_moves

        for line in perft_lines:
            line = line.strip()
            if not line:  # Skip any empty lines
                continue

            # Expected move format from 'go perft 1': "e2e4: 1", "e7e8q: 1", etc.
            # The final summary line is "Nodes searched: N"
            if "Nodes searched:" in line:
                continue  # This is the summary line, not a move

            parts = line.split(':', 1)  # Split on the first colon only
            if len(parts) == 2:
                move_uci_candidate = parts[0].strip()
                # Basic validation for a UCI move string (e.g., e2e4, e7e8q)
                # It should be 4 characters for a regular move, or 5 for a promotion.
                if 4 <= len(move_uci_candidate) <= 5:
                    uci_moves.append(move_uci_candidate)
                # else:
                    # Optional: log if a line with ':' didn't look like a move
                    # print(f"DEBUG: Discarding potential non-move line from perft output: '{line}'")
            # else:
                # Optional: log if a line didn't have ':' and wasn't "Nodes searched:"
                # print(f"DEBUG: Skipping unexpected line format in perft output: '{line}'")
        
        return uci_moves

    # This is the new public method incorporating Step 2.2 and Step 2.3
    def get_legal_moves_for_fen(self, fen: str) -> list[str] | None:
        """
        Retrieves a list of legal moves in UCI format for a given FEN position
        by communicating with the Stockfish engine.

        This method combines fetching raw engine output (Step 2.2) and parsing it (Step 2.3).

        Args:
            fen (str): The FEN string of the position, or "startpos" for the starting position.

        Returns:
            list[str] | None: A list of legal moves in UCI format (e.g., ["e2e4", "g1f3"]).
                              Returns None if communication with the engine fails at a critical step.
                              Returns an empty list if the position is terminal (no legal moves)
                              but communication was successful.
        """
        if not self.is_process_alive():
            print("ERROR (get_legal_moves_for_fen): Stockfish process not running.")
            return None # Indicates critical failure

        # print(f"INFO: Getting legal moves for FEN: '{fen}'") # Can be noisy
        
        # Step 2.2: Get raw output from the engine
        raw_output_lines = self.get_legal_moves_output(fen) 

        if raw_output_lines is None:
            # get_legal_moves_output method would have already printed an error
            print(f"ERROR (get_legal_moves_for_fen): Failed to get raw output from engine for FEN '{fen}'. Cannot parse moves.")
            return None # Indicates critical failure

        # Step 2.3: Parse the raw output for UCI moves
        parsed_uci_moves = self._parse_perft_output_for_uci_moves(raw_output_lines)

        # Check for terminal positions (e.g., checkmate/stalemate)
        # 'go perft 1' on a terminal position usually outputs "Nodes searched: 0" (or 1 for the root)
        # and no actual move lines. In this case, parsed_uci_moves would be empty.
        if not parsed_uci_moves and raw_output_lines:
            is_likely_terminal_output = any("Nodes searched:" in line for line in raw_output_lines)
            
            if is_likely_terminal_output:
                # print(f"INFO: No legal moves parsed for FEN '{fen}'. This is expected for a terminal position (e.g., checkmate/stalemate).")
                return [] # Correctly return an empty list for terminal positions
            else:
                # This is an unusual case: raw output was received, but nothing parsed,
                # and it doesn't look like the standard "Nodes searched" only output.
                print(f"WARNING (get_legal_moves_for_fen): Raw output received for FEN '{fen}', but no moves parsed and not clearly terminal. Output was:")
                for line_idx, line_content in enumerate(raw_output_lines):
                    print(f"  RAW[{line_idx+1}] > {line_content}")
                # Depending on desired strictness, could return None or [] here.
                # Let's return the (empty) parsed_uci_moves for now.
        
        # print(f"INFO: Parsed {len(parsed_uci_moves)} legal moves for FEN '{fen}'.")
        return parsed_uci_moves        