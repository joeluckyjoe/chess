import os
import shutil
from chess_environment import ChessEnvironmentInterface

def find_stockfish():
    """Helper function to find the Stockfish executable."""
    # Start with common names/paths or check PATH
    paths_to_try = ["stockfish", "/usr/games/stockfish", "/usr/bin/stockfish"]
    
    # Check if stockfish is in PATH first
    found_path = shutil.which("stockfish")
    if found_path:
        print(f"Found Stockfish in PATH: {found_path}")
        return found_path
        
    # If not in PATH, check some common locations
    for path in paths_to_try:
        if os.path.exists(path) and os.access(path, os.X_OK):
            print(f"Found Stockfish at: {path}")
            return path

    # Add a specific Windows path check if needed
    win_path = r"C:\path\to\stockfish\stockfish.exe" # <-- ADJUST THIS IF ON WINDOWS
    if os.name == 'nt' and os.path.exists(win_path) and os.access(win_path, os.X_OK):
         print(f"Found Stockfish at: {win_path}")
         return win_path
         
    return None # Not found

if __name__ == "__main__":
    stockfish_exe = find_stockfish()
    if not stockfish_exe:
        print("ERROR: Could not find Stockfish executable.")
        print("Please ensure Stockfish is installed and either in your system's PATH")
        print("or update the 'find_stockfish' function in this script with the correct path.")
        exit()

    env = None
    try:
        # Initialize the environment
        env = ChessEnvironmentInterface(stockfish_path=stockfish_exe)

        # 1. Test starting position
        print("\n--- Testing Starting Position ---")
        start_fen = env.get_current_fen()
        print(f"Starting FEN: {start_fen}")
        start_moves = env.get_legal_moves()
        print(f"Legal moves ({len(start_moves)}): {start_moves[:5]} ...")
        assert len(start_moves) == 20

        # 2. Apply a move (e.g., e4)
        print("\n--- Applying move e2e4 ---")
        move_to_apply = "e2e4"
        if move_to_apply in start_moves:
            env.apply_move(move_to_apply)
            e4_fen = env.get_current_fen()
            print(f"FEN after e2e4: {e4_fen}")
            e4_moves = env.get_legal_moves()
            print(f"Legal moves ({len(e4_moves)}): {e4_moves[:5]} ...")
            assert len(e4_moves) == 20 # After 1. e4, Black has 20 moves
            assert "e7e5" in e4_moves
        else:
            print(f"ERROR: Move {move_to_apply} not found in legal moves!")

        # 3. Apply another move (e.g., e5)
        print("\n--- Applying move e7e5 ---")
        move_to_apply_2 = "e7e5"
        if move_to_apply_2 in e4_moves:
            env.apply_move(move_to_apply_2)
            e5_fen = env.get_current_fen()
            print(f"FEN after e7e5: {e5_fen}")
            e5_moves = env.get_legal_moves()
            print(f"Legal moves ({len(e5_moves)}): {e5_moves[:5]} ...")
            assert len(e5_moves) == 29 # After 1. e4 e5, White has 29 moves.
            assert "g1f3" in e5_moves
        else:
             print(f"ERROR: Move {move_to_apply_2} not found in legal moves!")

        # 4. Test game over (should be false)
        print(f"\nGame Over? {env.is_game_over()}")
        assert not env.is_game_over()

        print("\n--- Test sequence completed successfully! ---")

    except RuntimeError as e:
        print(f"\n--- A runtime error occurred during testing: {e} ---")
    except Exception as e:
        print(f"\n--- An unexpected error occurred during testing: {e} ---")
        import traceback
        traceback.print_exc()
    finally:
        # 5. Close the environment
        if env:
            env.close()