import os
import torch
import pandas as pd
from pathlib import Path
import chess.pgn
import datetime
import subprocess
import sys
import re
import argparse # Added for command-line arguments

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
    most recently modified file. This is the most robust way to ensure
    we get the absolute latest state, especially after saving a temporary
    checkpoint.

    Args:
        checkpoints_path (Path): The directory where checkpoints are stored.

    Returns:
        Path: The Path object of the latest checkpoint, or None if no checkpoints are found.
    """
    if not checkpoints_path.is_dir():
        print(f"[Warning] Checkpoint directory not found at: {checkpoints_path}")
        return None

    files = list(checkpoints_path.glob('*.pth.tar'))
    if not files:
        return None

    # Find the file with the most recent modification time.
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


def main():
    """
    Main training loop that orchestrates self-play, mentor-play, and network training,
    guided by the new BayesianSupervisor.
    """
    parser = argparse.ArgumentParser(description="Run the MCTS RL training loop.")
    parser.add_argument(
        '--force-start-game', 
        type=int, 
        default=None,
        help="Force the training to start from a specific game number, ignoring the latest checkpoint number."
    )
    args = parser.parse_args()

    # --- 1. Get Environment-Aware Paths & Config ---
    paths = get_paths()
    checkpoints_path = paths.checkpoints_dir
    training_data_path = paths.training_data_dir
    pgn_path = paths.pgn_games_dir
    
    drive_root = paths.drive_project_root
    loss_log_filepath = drive_root / 'loss_log_v2.csv'
    supervisor_log_filepath = drive_root / 'supervisor_log.txt'
    
    local_root = paths.local_project_root

    device_str = config_params['DEVICE']
    device = "cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu"

    print(f"Using device: {device}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")
    print(f"Training data will be saved to: {training_data_path}")
    print(f"PGN files will be saved to: {pgn_path}")
    print(f"Log files will be saved to: {drive_root}")

    # --- 2. Initialize Components ---
    trainer = Trainer(
        model_config=config_params, 
        learning_rate=config_params['LEARNING_RATE'], 
        weight_decay=config_params['WEIGHT_DECAY'], 
        device=device
    )

    # --- 3. Load Checkpoint or Initialize Network ---
    print("Attempting to load the latest checkpoint...")
    
    checkpoint_to_load = None
    # If forcing a start, find the correct checkpoint to load for that state
    if args.force_start_game is not None:
        print("\n" + "*"*60)
        print(f"COMMAND-LINE OVERRIDE: Forcing start from game {args.force_start_game}.")
        print("*"*60 + "\n")
        
        # Helper to find the latest checkpoint at or before the forced start game
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

    # The Trainer will now handle loading the specific checkpoint if provided, or the latest otherwise
    chess_network, start_game = trainer.load_or_initialize_network(checkpoints_path, specific_checkpoint_path=checkpoint_to_load)

    # If we forced a start, override the game number returned by the loader
    if args.force_start_game is not None:
        start_game = args.force_start_game - 1

    if start_game > 0:
        print(f"Resuming training from game {start_game + 1}")
    else:
        print("Starting training from scratch.")
    
    # --- 4. Initialize Players & Supervisor ---
    mcts_player = MCTS(
        network=chess_network, 
        device=device, 
        c_puct=config_params['CPUCT']
    )
    
    self_player = SelfPlay(
        mcts_white=mcts_player, 
        mcts_black=mcts_player, 
        stockfish_path=config_params['STOCKFISH_PATH'],
        num_simulations=config_params['MCTS_SIMULATIONS']
    )
    
    mentor_elo = config_params.get('MENTOR_ELO_RATING', 1350)

    mentor_player = MentorPlay(
        mcts_agent=mcts_player,
        stockfish_path=config_params['STOCKFISH_PATH'],
        stockfish_elo=mentor_elo,
        num_simulations=config_params['MCTS_SIMULATIONS'],
        agent_color_str=config_params['MENTOR_GAME_AGENT_COLOR']
    )

    training_data_manager = TrainingDataManager(data_directory=training_data_path)

    print("Initializing Bayesian Supervisor...")
    supervisor = BayesianSupervisor(config=config_params)
    
    last_mode, _ = get_last_game_info(loss_log_filepath)
    current_mode = last_mode
    print(f"Initialized mode based on last game in log: '{current_mode}'")

    # --- 5. Main Training Loop ---
    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        
        # =====================================================================
        # TACTICS TRAINING INTERVENTION
        # =====================================================================
        if game_num > 1 and game_num % config_params['TACTICS_SESSION_FREQUENCY'] == 0:
            print("\n" + "="*80)
            print(f"--- GAME {game_num}: INITIATING TACTICS TRAINING SESSION ---")
            print("="*80)

            pre_tactics_checkpoint_name = f"checkpoint_game_{game_num - 1}_pre_tactics.pth.tar"
            print(f"Saving temporary pre-tactics checkpoint: {pre_tactics_checkpoint_name}")
            trainer.save_checkpoint(
                directory=checkpoints_path, 
                game_number=game_num - 1,
                filename_override=pre_tactics_checkpoint_name
            )

            tactics_script_path = local_root / 'train_on_tactics.py'
            puzzles_file_path = paths.tactical_puzzles_file

            if not puzzles_file_path.exists():
                print(f"[WARNING] Tactical puzzles file not found at '{puzzles_file_path}'. Skipping tactics session.")
            else:
                # Find the latest checkpoint by modification time to get the one we just saved
                latest_checkpoint_path = find_latest_checkpoint_by_time(checkpoints_path)
                
                if not latest_checkpoint_path:
                    print("[WARNING] No checkpoint found to train on. Skipping tactics session.")
                else:
                    print(f"Found latest checkpoint for tactics training: {latest_checkpoint_path.name}")
                    command = [
                        sys.executable, str(tactics_script_path),
                        '--puzzles_path', str(puzzles_file_path),
                        '--model_path', str(latest_checkpoint_path)
                    ]
                    
                    try:
                        subprocess.run(command, check=True)
                        print("Tactics training subprocess finished. Reloading the updated model...")
                        # Reload the absolute latest model after tactics training
                        chess_network, _ = trainer.load_or_initialize_network(checkpoints_path)
                        mcts_player.network = chess_network
                        print("Successfully reloaded tactics-trained model into the current session.")

                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] The tactics training script failed with exit code {e.returncode}. Continuing with the old model.")
                    except Exception as e:
                        print(f"[ERROR] An unexpected error occurred during tactics training: {e}. Continuing with the old model.")
            
            print("\n" + "="*80)
            print("--- TACTICS SESSION COMPLETE ---")
            print("="*80 + "\n")


        previous_mode = current_mode
        
        log_df = pd.read_csv(loss_log_filepath) if os.path.exists(loss_log_filepath) else pd.DataFrame(columns=['game', 'game_type'])
        mentor_games = log_df[log_df['game_type'] == 'mentor-play']
        last_mentor_game = mentor_games['game'].max() if not mentor_games.empty else -1
        
        games_since_mentor = game_num - last_mentor_game
        
        if last_mentor_game != -1 and games_since_mentor <= config_params['SUPERVISOR_GRACE_PERIOD']:
            print(f"Within grace period ({games_since_mentor}/{config_params['SUPERVISOR_GRACE_PERIOD']} games since mentor). Forcing self-play.")
            current_mode = "self-play"
        else:
            stagnation_detected = supervisor.check_for_stagnation(loss_log_filepath)
            if stagnation_detected:
                current_mode = "mentor-play"
            else:
                current_mode = "self-play"
        
        if current_mode != previous_mode:
            reason_for_switch = f"Supervisor switched mode from '{previous_mode}' to '{current_mode}'."
            print(reason_for_switch)
            with open(supervisor_log_filepath, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{timestamp}] Game {game_num}: {reason_for_switch}\n"
                f.write(log_message)
        
        print(f"\n--- Game {game_num}/{config_params['TOTAL_GAMES']} (Mode: {current_mode}) ---")
        
        training_examples = []
        pgn_data = None
        
        if current_mode == "mentor-play":
            training_examples, pgn_data = mentor_player.play_game()
        else: # self-play
            training_examples, pgn_data = self_player.play_game()

        if not training_examples:
            print(f"Game type '{current_mode}' resulted in no training examples. Skipping.")
            continue
        
        num_moves = len(list(pgn_data.mainline_moves())) if pgn_data else 0
        print(f"{current_mode.capitalize()} game complete ({num_moves} moves). Generated {len(training_examples)} examples.")

        data_filename = f"{current_mode}_game_{game_num}_data.pkl"
        training_data_manager.save_data(training_examples, filename=data_filename)
        
        if pgn_data:
            pgn_filename = pgn_path / f"{current_mode}_game_{game_num}.pgn"
            try:
                with open(pgn_filename, "w", encoding="utf-8") as f:
                    print(pgn_data, file=f, end="\n\n")
                print(f"Successfully saved PGN to {pgn_filename}")
            except Exception as e:
                print(f"[ERROR] Could not save PGN file: {e}")

        print(f"Training on the {len(training_examples)} examples from game {game_num}...")
        final_policy_loss = 0
        final_value_loss = 0
        for epoch in range(config_params['TRAINING_EPOCHS']):
            policy_loss, value_loss = trainer.train_on_batch(training_examples, batch_size=config_params['BATCH_SIZE'])
            if epoch == config_params['TRAINING_EPOCHS'] - 1:
                final_policy_loss = policy_loss
                final_value_loss = value_loss
            print(f"Epoch {epoch + 1}/{config_params['TRAINING_EPOCHS']} complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        write_loss_to_csv(loss_log_filepath, game_num, final_policy_loss, final_value_loss, current_mode)

        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=checkpoints_path, game_number=game_num)

    print("\n--- Training Run Finished ---")
    self_player.close()
    mentor_player.close()

if __name__ == "__main__":
    main()
