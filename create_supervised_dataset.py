import torch
import chess
import chess.pgn
from pathlib import Path
import sys
from collections import deque
from tqdm import tqdm

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))

from config import get_paths
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from hardware_setup import get_device

# --- Configuration ---
# Set to None to process all games in the corpus.
MAX_GAMES_TO_PROCESS = None
# The sequence length our temporal model expects
SEQUENCE_LENGTH = 8

def get_value_target(result: str, player_turn: chess.Color) -> float:
    if result == '1-0':
        return 1.0 if player_turn == chess.WHITE else -1.0
    elif result == '0-1':
        return -1.0 if player_turn == chess.WHITE else 1.0
    else:
        return 0.0

def main():
    print("--- Starting Supervised Dataset Generation ---")
    paths = get_paths()
    device = get_device()
    
    pgn_corpus_dir = paths.drive_project_root / 'pgn_corpus'
    output_dir = paths.drive_project_root / 'training_data'
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    pgn_files = sorted(list(pgn_corpus_dir.glob('*.pgn'))) # Sort for consistent order
    if not pgn_files:
        print(f"[FATAL] No PGN files found in {pgn_corpus_dir}")
        sys.exit(1)

    print(f"Found {len(pgn_files)} PGN files to process.")
    total_games_processed = 0

    empty_board = chess.Board()
    initial_gnn, initial_cnn, _ = convert_to_gnn_input(empty_board, device)
    initial_state_tuple = (initial_gnn, initial_cnn)

    for pgn_file_path in pgn_files:
        # <<< MODIFIED: Create a unique output path for each input file.
        output_path = output_dir / f"{pgn_file_path.stem}.pt"
        
        # <<< MODIFIED: This is the core of the resume logic.
        if output_path.exists():
            print(f"\nSkipping already processed file: {pgn_file_path.name}")
            continue

        print(f"\nProcessing file: {pgn_file_path.name}")
        
        file_training_data = []
        games_in_file_count = 0

        with open(pgn_file_path, encoding='utf-8', errors='ignore') as pgn_file:
            # We can't easily get a total count, so the progress bar will just count games.
            with tqdm(desc=f"Processing {pgn_file_path.name}") as pbar:
                while True:
                    if MAX_GAMES_TO_PROCESS and total_games_processed >= MAX_GAMES_TO_PROCESS:
                        # Save any pending data before stopping completely
                        if file_training_data:
                           print(f"\nLimit reached. Saving final chunk for {pgn_file_path.name}...")
                           torch.save(file_training_data, output_path)
                        raise StopIteration

                    try:
                        game = chess.pgn.read_game(pgn_file)
                    except Exception:
                        continue
                        
                    if game is None:
                        break

                    games_in_file_count += 1
                    total_games_processed += 1
                    pbar.update(1)

                    result = game.headers.get("Result", "*")
                    if result == '*':
                        continue

                    board = game.board()
                    state_deque = deque([initial_state_tuple] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
                    
                    for move in game.mainline_moves():
                        player_turn = board.turn
                        policy_target = torch.zeros(get_action_space_size())
                        try:
                            action_index = move_to_index(move, board)
                            if action_index is not None:
                                policy_target[action_index] = 1.0
                            else:
                                continue
                        except Exception:
                            continue

                        value_target = torch.tensor([get_value_target(result, player_turn)], dtype=torch.float32)

                        file_training_data.append({
                            'state_sequence': list(state_deque),
                            'policy': policy_target,
                            'value_target': value_target
                        })

                        board.push(move)
                        gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
                        state_deque.append((gnn_data, cnn_tensor))
        
        if file_training_data:
            print(f"\nSaving data for {pgn_file_path.name} ({len(file_training_data)} samples)...")
            torch.save(file_training_data, output_path)
            print(f"✅ Saved to {output_path}")

    print("\n--- All files processed. Dataset generation complete. ---")


if __name__ == "__main__":
    main()