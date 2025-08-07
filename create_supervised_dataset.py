import torch
import chess
import chess.pgn
from pathlib import Path
import sys
from collections import deque
from tqdm import tqdm
import re

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))

from config import get_paths
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size, move_to_index
from hardware_setup import get_device

# --- Configuration ---
SAVE_CHECKPOINT_INTERVAL = 50000  # Save a new chunk every 50,000 samples
SEQUENCE_LENGTH = 8
AVG_SAMPLES_PER_GAME = 90 # An estimated average for calculating games to skip

def get_value_target(result: str, player_turn: chess.Color) -> float:
    if result == '1-0': return 1.0 if player_turn == chess.WHITE else -1.0
    elif result == '0-1': return -1.0 if player_turn == chess.WHITE else 1.0
    return 0.0

def find_last_checkpoint(output_dir):
    """Finds the last saved chunk number and the total samples processed."""
    max_chunk_num = 0
    total_samples = 0
    for f in output_dir.glob("supervised_dataset_part_*.pt"):
        match = re.search(r'part_(\d+)_(\d+)_samples', f.name)
        if match:
            chunk_num = int(match.group(1))
            samples_in_chunk = int(match.group(2))
            if chunk_num > max_chunk_num:
                max_chunk_num = chunk_num
            total_samples += samples_in_chunk
    return max_chunk_num, total_samples

def main():
    print("--- Starting Supervised Dataset Generation (Resumable) ---")
    paths = get_paths()
    device = get_device()
    
    pgn_corpus_dir = paths.drive_project_root / 'pgn_corpus'
    output_dir = paths.drive_project_root / 'training_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    pgn_files = sorted(list(pgn_corpus_dir.glob('*.pgn')))
    if not pgn_files:
        print(f"[FATAL] No PGN files found in {pgn_corpus_dir}")
        sys.exit(1)

    # <<< MODIFIED: More efficient resume logic >>>
    last_chunk_num, total_samples_processed = find_last_checkpoint(output_dir)
    games_to_skip = 0
    if last_chunk_num > 0:
        games_to_skip = total_samples_processed // AVG_SAMPLES_PER_GAME
        print(f"Resuming after chunk {last_chunk_num}. Skipping approximately {games_to_skip} games...")
    
    checkpoint_counter = last_chunk_num
    current_training_data = []
    samples_since_last_save = 0
    
    empty_board = chess.Board()
    initial_gnn, initial_cnn, _ = convert_to_gnn_input(empty_board, device)
    initial_state_tuple = (initial_gnn, initial_cnn)

    try:
        for pgn_file_path in pgn_files:
            print(f"\nProcessing file: {pgn_file_path.name}")
            with open(pgn_file_path, encoding='utf-8', errors='ignore') as pgn_file:
                with tqdm(desc=f"Processing {pgn_file_path.name}", unit="game") as pbar:
                    while True:
                        try:
                            game = chess.pgn.read_game(pgn_file)
                        except Exception:
                            continue
                        if game is None: break

                        pbar.update(1)

                        # <<< MODIFIED: Skip entire games, which is much faster >>>
                        if games_to_skip > 0:
                            games_to_skip -= 1
                            continue

                        result = game.headers.get("Result", "*")
                        if result == '*': continue

                        board = game.board()
                        state_deque = deque([initial_state_tuple] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
                        
                        for move in game.mainline_moves():
                            player_turn = board.turn
                            policy_target = torch.zeros(get_action_space_size())
                            try:
                                action_index = move_to_index(move, board)
                                if action_index is None: continue
                                policy_target[action_index] = 1.0
                            except Exception:
                                continue

                            value_target = torch.tensor([get_value_target(result, player_turn)], dtype=torch.float32)

                            current_training_data.append({
                                'state_sequence': list(state_deque), 'policy': policy_target, 'value_target': value_target
                            })
                            samples_since_last_save += 1

                            board.push(move)
                            gnn_data, cnn_tensor, _ = convert_to_gnn_input(board, device)
                            state_deque.append((gnn_data, cnn_tensor))
                            
                            if samples_since_last_save >= SAVE_CHECKPOINT_INTERVAL:
                                checkpoint_counter += 1
                                num_samples_in_chunk = len(current_training_data)
                                save_path = output_dir / f"supervised_dataset_part_{checkpoint_counter}_{num_samples_in_chunk}_samples.pt"
                                print(f"\nSaving chunk {checkpoint_counter} with {num_samples_in_chunk} samples to {save_path}")
                                torch.save(current_training_data, save_path)
                                current_training_data = []
                                samples_since_last_save = 0
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if current_training_data:
            checkpoint_counter += 1
            num_samples_in_chunk = len(current_training_data)
            save_path = output_dir / f"supervised_dataset_part_{checkpoint_counter}_{num_samples_in_chunk}_samples.pt"
            print(f"\nSaving final chunk {checkpoint_counter} with {num_samples_in_chunk} samples to {save_path}")
            torch.save(current_training_data, save_path)
        
        print("\n✅ Dataset generation finished.")

if __name__ == "__main__":
    main()