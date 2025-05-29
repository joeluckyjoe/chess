import os  # <--- Add this line
import shutil # This was likely already there from the example
from stockfish_communicator import StockfishCommunicator

if __name__ == '__main__':
    # IMPORTANT: Replace this with the actual path to your Stockfish executable
    # Examples:
    # stockfish_executable_path = "/usr/games/stockfish"  # Linux example
    # stockfish_executable_path = r"C:\Users\YourUser\Desktop\stockfish_16_win_x64_avx2\stockfish-windows-x86-64-avx2.exe" # Windows example
    stockfish_executable_path = "stockfish" # If stockfish is in your system's PATH

    # Attempt to find stockfish if a direct path isn't immediately known or if it's in PATH
    import shutil
    if not os.path.exists(stockfish_executable_path) or not os.access(stockfish_executable_path, os.X_OK) :
        found_path = shutil.which("stockfish")
        if found_path:
            print(f"Using Stockfish found in PATH: {found_path}")
            stockfish_executable_path = found_path
        else:
            print(f"ERROR: Stockfish not found at '{stockfish_executable_path}' and not in system PATH.")
            print("Please install Stockfish and provide the correct path.")
            exit()

    communicator = None
    try:
        # This is where stockfish_executable_path should have been determined
        print(f"INFO: Determined Stockfish path: '{stockfish_executable_path}'")
        print("INFO: Attempting to instantiate StockfishCommunicator...")
        
        communicator = StockfishCommunicator(stockfish_path=stockfish_executable_path) # Instantiation attempt
        
        print("INFO: StockfishCommunicator instantiated successfully.")

        # ... (inside the try block, after successful handshake)
        if communicator.perform_handshake():
            print("\n✅ Handshake successful with Stockfish.")
            # ... (previous print message) ...

            # --- Test for Step 2.3 (integrates Step 2.2 and 2.3) ---
            print("\n--- Testing Get Parsed Legal Moves (Step 2.3) ---")
            
            # Test Case 1: Standard starting position
            fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            print(f"\nAttempting to get parsed legal moves for FEN (startpos): '{fen_start}'")
            legal_moves_start = communicator.get_legal_moves_for_fen(fen_start)

            if legal_moves_start is not None:
                print(f"SUCCESS: Parsed {len(legal_moves_start)} legal moves for start FEN.")
                if legal_moves_start: # If list is not empty
                    print(f"  First 5 moves: {legal_moves_start[:5]}")
                    if len(legal_moves_start) > 5:
                        print(f"  Total moves: {len(legal_moves_start)}")
                else: # List is empty
                    print("  Received an empty list of moves (expected for terminal positions, but not for startpos).")
            else:
                print(f"FAILED: Could not get parsed legal moves for start FEN (communication or major parsing error).")

            # Test Case 2: A position with fewer moves (e.g., an endgame)
            fen_endgame = "8/8/4k3/8/8/8/4K3/5R2 w - - 0 1" # White to move, King and Rook vs King
            print(f"\nAttempting to get parsed legal moves for FEN (endgame): '{fen_endgame}'")
            legal_moves_endgame = communicator.get_legal_moves_for_fen(fen_endgame)
            if legal_moves_endgame is not None:
                print(f"SUCCESS: Parsed {len(legal_moves_endgame)} legal moves for endgame FEN: {legal_moves_endgame}")
            else:
                print(f"FAILED: Could not get parsed legal moves for endgame FEN.")

            # Test Case 3: A checkmated position (should have no legal moves)
            # Fool's mate variation: 1.f3 e5 2.g4 Qh4#
            fen_checkmate = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3" # White is checkmated
            print(f"\nAttempting to get parsed legal moves for FEN (checkmate): '{fen_checkmate}'")
            legal_moves_checkmate = communicator.get_legal_moves_for_fen(fen_checkmate)

            if legal_moves_checkmate is not None:
                if not legal_moves_checkmate: # Empty list is expected
                    print(f"SUCCESS: Parsed {len(legal_moves_checkmate)} legal moves (empty list) for checkmate FEN. This is correct.")
                else: # Moves found, which would be incorrect for this FEN
                    print(f"UNEXPECTED: Parsed {len(legal_moves_checkmate)} legal moves for checkmate FEN: {legal_moves_checkmate}")
            else:
                print(f"FAILED: Could not get parsed legal moves for checkmate FEN.")
            # --- End of Test for Step 2.3 ---

        else:
            print("\n❌ Handshake failed (perform_handshake returned False).")

    except FileNotFoundError as e_init:
        print(f"ERROR DURING INIT (FileNotFoundError): {e_init}")
        import traceback
        traceback.print_exc()
    except PermissionError as e_init:
        print(f"ERROR DURING INIT (PermissionError): {e_init}")
        import traceback
        traceback.print_exc()
    except Exception as e_general:
        # This block will catch the original error if instantiation failed,
        # or errors from perform_handshake/get_legal_moves_output if instantiation succeeded.
        print(f"--- AN UNEXPECTED ERROR OCCURRED ---")
        if communicator is None:
            print(f"ERROR: StockfishCommunicator instantiation likely failed.")
        print(f"Error Type: {type(e_general).__name__}")
        print(f"Error Message: {e_general}")
        print("--- Traceback ---")
        import traceback
        traceback.print_exc()
        print("-----------------")
    finally:
        if communicator and communicator.is_process_alive(): # Check if communicator exists and process is alive
            print("INFO: Closing Stockfish communicator in finally block...")
            communicator.close()
        elif communicator: # Communicator exists but process might be dead
             print("INFO: Stockfish process was not alive or communicator exists but process dead, ensuring cleanup in finally...")
             communicator.close() # Attempt close anyway to clean up threads etc.
        else:
            print("INFO: Communicator is None in finally block (was not successfully initialized).")