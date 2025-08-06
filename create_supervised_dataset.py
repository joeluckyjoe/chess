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

# Save data in chunks to prevent memory overload.
SAVE_CHECKPOINT_INTERVAL = 100000

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
    if not pgn_corpus_dir.exists():
        print(f"[FATAL] PGN Corpus directory not found at: {pgn_corpus_dir}")
        sys.exit(1)

    pgn_files = list(pgn_corpus_dir.glob('*.pgn'))
    if not pgn_files:
        print(f"[FATAL] No PGN files found in {pgn_corpus_dir}")
        sys.exit(1)

    print(f"Found {len(pgn_files)} PGN files to process.")

    all_training_data = []
    games_processed_count = 0
    samples_since_last_save = 0
    checkpoint_counter = 0

    empty_board = chess.Board()
    initial_gnn, initial_cnn, _ = convert_to_gnn_input(empty_board, device)
    initial_state_tuple = (initial_gnn, initial_cnn)

    try:
        for pgn_file_path in pgn_files:
            print(f"\nProcessing file: {pgn_file_path.name}")
            with open(pgn_file_path, encoding='utf-8', errors='ignore') as pgn_file:
                with tqdm(desc="Processing games") as pbar:
                    while True:
                        # <<< FIXED: Corrected the variable name's casing
                        if MAX_GAMES_TO_PROCESS and games_processed_count >= MAX_GAMES_TO_PROCESS:
                            raise StopIteration

                        try:
                            game = chess.pgn.read_game(pgn_file)
                        except (ValueError, UnicodeDecodeError) as e:
                            pbar.set_postfix_str(f"Skipping bad game: {e}")
                            continue
                            
                        if game is None:
                            break

                        games_processed_count += 1
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

                            all_training_data.append({
                                'state_sequence': list(state_deque),
                                'policy': policy_target,
                                'value_target': value_target
                            })
                            samples_since_last_save += 1

                            board.push(move)
                            gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
                            state_deque.append((gnn_data, cnn_tensor))
                            
                            if samples_since_last_save >= SAVE_CHECKPOINT_INTERVAL:
                                checkpoint_counter += 1
                                save_path = paths.drive_project_root / 'training_data' / f"supervised_dataset_part_{checkpoint_counter}.pt"
                                save_path.parent.mkdir(parents=True, exist_ok=True)
                                print(f"\nSaving chunk {checkpoint_counter} with {len(all_training_data)} samples to {save_path}")
                                torch.save(all_training_data, save_path)
                                all_training_data = [] # Clear memory
                                samples_since_last_save = 0

    except StopIteration:
        print(f"\nReached the limit of {MAX_GAMES_TO_PROCESS} games.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    finally:
        if all_training_data:
            checkpoint_counter += 1
            save_path = paths.drive_project_root / 'training_data' / f"supervised_dataset_part_{checkpoint_counter}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving final chunk {checkpoint_counter} with {len(all_training_data)} samples to {save_path}")
            torch.save(all_training_data, save_path)
        
        print("\n✅ Dataset generation finished.")


if __name__ == "__main__":
    main()