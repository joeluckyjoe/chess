# FILENAME: run_training.py
import os
import torch
import pandas as pd
from pathlib import Path
import chess.pgn
import datetime
import subprocess
import sys
import re
import argparse
import json

# --- Import from config ---
from config import get_paths, config_params

# Core components from the gnn_agent package
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.search.mcts import MCTS
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.mentor_play import MentorPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer
# --- Import the BayesianSupervisor ---
from gnn_agent.rl_loop.bayesian_supervisor import BayesianSupervisor


def find_latest_checkpoint_by_time(checkpoints_path: Path):
    """
    Finds the latest checkpoint file in the given directory by finding the
    most recently modified file.
    """
    if not checkpoints_path.is_dir():
        print(f"[Warning] Checkpoint directory not found at: {checkpoints_path}")
        return None
    files = list(checkpoints_path.glob('*.pth.tar'))
    if not files:
        return None
    latest_checkpoint_path = max(files, key=os.path.getmtime)
    return latest_checkpoint_path


def write_loss_to_csv(filepath, game_num, policy_loss, value_loss, game_type):
    """Appends a new row of loss data to a CSV file."""
    file_exists = os.path.isfile(filepath)
    df = pd.DataFrame([[game_num, policy_loss, value_loss, game_type]], columns=['game', 'policy_loss', 'value_loss', 'game_type'])
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)


def get_last_game_info(log_file_path):
    """Reads the log file to determine the mode and game number of the last completed game."""
    if not os.path.exists(log_file_path):
        return "self-play", 0
    try:
        df = pd.read_csv(log_file_path)
        if df.empty:
            return "self-play", 0
        last_row = df.iloc[-1]
        return last_row['game_type'], last_row['game']
    except (pd.errors.EmptyDataError, IndexError):
        return "self-play", 0

# --- PHASE T MODIFICATION: ADDED ---
def load_tactical_puzzles(puzzles_path: Path):
    """Loads tactical puzzles from a .jsonl file."""
    if not puzzles_path.exists():
        print(f"[WARNING] Tactical puzzles file not found at {puzzles_path}. Continuing without them.")
        return []
    puzzles = []
    with open(puzzles_path, 'r') as f:
        for line in f:
            puzzles.append(json.loads(line))
    print(f"Successfully loaded {len(puzzles)} tactical puzzles.")
    return puzzles
# --- END MODIFICATION ---

def main():
    """
    Main training loop that orchestrates self-play, mentor-play, and network training.
    """
    parser = argparse.ArgumentParser(description="Run the MCTS RL training loop.")
    parser.add_argument(
        '--force-start-game',
        type=int,
        default=None,
        help="Force the training to start from a specific game number, ignoring the latest checkpoint number."
    )
    args = parser.parse_args()

    paths = get_paths()
    checkpoints_path = paths.checkpoints_dir
    training_data_path = paths.training_data_dir
    pgn_path = paths.pgn_games_dir
    drive_root = paths.drive_project_root
    loss_log_filepath = drive_root / 'loss_log_v2.csv'
    supervisor_log_filepath = drive_root / 'supervisor_log.txt'
    
    # --- PHASE T MODIFICATION: ADDED ---
    tactical_puzzles_path = paths.tactical_puzzles_file
    # --- END MODIFICATION ---

    device_str = config_params['DEVICE']
    device = "cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu"

    print(f"Using device: {device}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")

    trainer = Trainer(
        model_config=config_params,
        learning_rate=config_params['LEARNING_RATE'],
        weight_decay=config_params['WEIGHT_DECAY'],
        device=device
    )

    print("Attempting to load checkpoint...")
    checkpoint_to_load = None
    if args.force_start_game is not None:
        print("\n" + "*"*60)
        print(f"COMMAND-LINE OVERRIDE: Forcing start from game {args.force_start_game}.")
        print("*"*60 + "\n")

        def find_checkpoint_for_game(max_game: int):
            max_game_num = -1
            path_to_load = None
            for f in checkpoints_path.glob('*.pth.tar'):
                match = re.search(r'_game_(\d+)', f.name)
                if match:
                    game_num = int(match.group(1))
                    if game_num <= max_game and game_num > max_game_num:
                        max_game_num = game_num
                        path_to_load = f
            return path_to_load
        checkpoint_to_load = find_checkpoint_for_game(args.force_start_game - 1)

    chess_network, start_game = trainer.load_or_initialize_network(
        checkpoints_path, specific_checkpoint_path=checkpoint_to_load
    )

    if args.force_start_game is not None:
        start_game = args.force_start_game - 1

    print(f"Resuming training from game {start_game + 1}")

    # --- PHASE T MODIFICATION: ADDED ---
    print("Loading tactical puzzles for integrated training...")
    tactical_puzzles = load_tactical_puzzles(tactical_puzzles_path)
    # --- END MODIFICATION ---

    mcts_player = MCTS(network=chess_network, device=device, c_puct=config_params['CPUCT'])
    self_player = SelfPlay(mcts_white=mcts_player, mcts_black=mcts_player, stockfish_path=config_params['STOCKFISH_PATH'], num_simulations=config_params['MCTS_SIMULATIONS'])
    mentor_player = MentorPlay(mcts_agent=mcts_player, stockfish_path=config_params['STOCKFISH_PATH'], stockfish_elo=config_params.get('MENTOR_ELO_RATING', 1350), num_simulations=config_params['MCTS_SIMULATIONS'], agent_color_str=config_params['MENTOR_GAME_AGENT_COLOR'])
    training_data_manager = TrainingDataManager(data_directory=training_data_path)
    supervisor = BayesianSupervisor(config=config_params)

    last_mode, _ = get_last_game_info(loss_log_filepath)
    current_mode = last_mode
    print(f"Initialized mode based on last game in log: '{current_mode}'")

    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        
        # --- PHASE T MODIFICATION: REMOVED ---
        # The entire periodic tactics training session block has been removed.
        # --- END MODIFICATION ---

        previous_mode = current_mode
        log_df = pd.read_csv(loss_log_filepath) if os.path.exists(loss_log_filepath) else pd.DataFrame(columns=['game', 'game_type'])
        mentor_games = log_df[log_df['game_type'] == 'mentor-play']
        last_mentor_game = mentor_games['game'].max() if not mentor_games.empty else -1
        games_since_mentor = game_num - last_mentor_game

        if last_mentor_game != -1 and games_since_mentor <= config_params['SUPERVISOR_GRACE_PERIOD']:
            current_mode = "self-play"
        else:
            current_mode = "mentor-play" if supervisor.check_for_stagnation(loss_log_filepath) else "self-play"
        
        if current_mode != previous_mode:
            print(f"Supervisor switched mode from '{previous_mode}' to '{current_mode}'.")
        
        print(f"\n--- Game {game_num}/{config_params['TOTAL_GAMES']} (Mode: {current_mode}) ---")
        
        if current_mode == "mentor-play":
            training_examples, pgn_data = mentor_player.play_game()
        else:
            training_examples, pgn_data = self_player.play_game()

        if not training_examples:
            print(f"Game type '{current_mode}' resulted in no examples. Skipping.")
            continue
        
        print(f"{current_mode.capitalize()} game complete. Generated {len(training_examples)} examples.")
        
        data_filename = f"{current_mode}_game_{game_num}_data.pkl"
        training_data_manager.save_data(training_examples, filename=data_filename)
        
        if pgn_data:
            pgn_filename = pgn_path / f"{current_mode}_game_{game_num}.pgn"
            try:
                with open(pgn_filename, "w", encoding="utf-8") as f:
                    print(pgn_data, file=f, end="\n\n")
            except Exception as e:
                print(f"[ERROR] Could not save PGN file: {e}")

        print(f"Training on {len(training_examples)} examples mixed with tactical puzzles...")
        
        # --- PHASE T MODIFICATION: MODIFIED ---
        # The trainer now receives the full list of puzzles to create a mixed batch.
        # The signature and logic inside trainer.train_on_batch will need to be updated.
        policy_loss, value_loss = trainer.train_on_batch(
            game_examples=training_examples,
            puzzle_examples=tactical_puzzles, # Pass the loaded puzzles
            batch_size=config_params['BATCH_SIZE']
        )
        # --- END MODIFICATION ---
        
        print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        write_loss_to_csv(loss_log_filepath, game_num, policy_loss, value_loss, current_mode)

        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=checkpoints_path, game_number=game_num)

    print("\n--- Training Run Finished ---")
    self_player.close()
    mentor_player.close()

if __name__ == "__main__":
    main()