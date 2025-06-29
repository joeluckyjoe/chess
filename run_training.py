import os
import torch
import pandas as pd
from pathlib import Path
import chess.pgn
import datetime
import subprocess # Added for calling tactics trainer
import sys # Added for getting python executable

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


def write_loss_to_csv(filepath, game_num, policy_loss, value_loss, game_type):
    """Appends a new row of loss data to a CSV file."""
    file_exists = os.path.isfile(filepath)
    df = pd.DataFrame([[game_num, policy_loss, value_loss, game_type]], columns=['game', 'policy_loss', 'value_loss', 'game_type'])
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)

def get_last_game_info(log_file_path):
    """Reads the log file to determine the mode and game number of the last completed game."""
    if not os.path.exists(log_file_path):
        return "self-play", 0 # Default if no log exists
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
    # --- 1. Get Environment-Aware Paths & Config ---
    paths = get_paths()
    checkpoints_path = paths.checkpoints_dir
    training_data_path = paths.training_data_dir
    pgn_path = paths.pgn_games_dir
    
    # MODIFICATION: Use the new persistent drive_project_root for logs
    drive_root = paths.drive_project_root
    loss_log_filepath = drive_root / 'loss_log_v2.csv'
    supervisor_log_filepath = drive_root / 'supervisor_log.txt'
    
    # This path is for the location of the code itself
    local_root = paths.local_project_root

    device_str = config_params['DEVICE']
    device = "cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu"

    print(f"Using device: {device}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")
    print(f"Training data will be saved to: {training_data_path}")
    print(f"PGN files will be saved to: {pgn_path}")
    print(f"Log files will be saved to: {drive_root}") # Added print for clarity

    # --- 2. Initialize Components ---
    trainer = Trainer(
        model_config=config_params, 
        learning_rate=config_params['LEARNING_RATE'], 
        weight_decay=config_params['WEIGHT_DECAY'], 
        device=device
    )

    # --- 3. Load Checkpoint or Initialize Network ---
    print("Attempting to load the latest checkpoint...")
    chess_network, start_game = trainer.load_or_initialize_network(checkpoints_path)
    
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
    
    mentor_player = MentorPlay(
        mcts_agent=mcts_player,
        stockfish_path=config_params['STOCKFISH_PATH'],
        stockfish_depth=config_params['STOCKFISH_DEPTH_MENTOR'],
        num_simulations=config_params['MCTS_SIMULATIONS'],
        agent_color_str=config_params['MENTOR_GAME_AGENT_COLOR']
    )

    training_data_manager = TrainingDataManager(data_directory=training_data_path)

    print("Initializing Bayesian Supervisor...")
    supervisor = BayesianSupervisor(config=config_params)
    
    # --- State Initialization with Memory ---
    last_mode, last_game_num = get_last_game_info(loss_log_filepath)
    current_mode = last_mode
    print(f"Initialized mode based on last game in log: '{current_mode}'")

    # --- 5. Main Training Loop ---
    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        
        # =====================================================================
        # TACTICS TRAINING INTERVENTION (Phase O)
        # =====================================================================
        if game_num > 1 and game_num % config_params['TACTICS_SESSION_FREQUENCY'] == 0:
            print("\n" + "="*80)
            print(f"--- GAME {game_num}: INITIATING TACTICS TRAINING SESSION ---")
            print("="*80)

            # Define the path to the tactics trainer script
            tactics_script_path = local_root / 'train_on_tactics.py'
            puzzles_file_path = paths.tactical_puzzles_file

            if not puzzles_file_path.exists():
                print(f"[WARNING] Tactical puzzles file not found at '{puzzles_file_path}'. Skipping tactics session.")
            else:
                # Find the current latest checkpoint to pass to the trainer
                latest_checkpoint_path = trainer.find_latest_checkpoint(checkpoints_path)
                if not latest_checkpoint_path:
                    print("[WARNING] No checkpoint found to train on. Skipping tactics session.")
                else:
                    # Run the training script as a subprocess
                    command = [
                        sys.executable, str(tactics_script_path),
                        '--puzzles_path', str(puzzles_file_path),
                        '--model_path', str(latest_checkpoint_path)
                    ]
                    
                    try:
                        subprocess.run(command, check=True)
                        # CRUCIAL: After training, we must reload the newly saved model
                        print("Tactics training subprocess finished. Reloading the updated model...")
                        # The new model will be named "..._tactics_trained.pth.tar"
                        tactics_trained_model_path = checkpoints_path / f"{latest_checkpoint_path.stem}_tactics_trained.pth.tar"
                        if tactics_trained_model_path.exists():
                            chess_network, _ = trainer.load_or_initialize_network(checkpoints_path, specific_checkpoint=tactics_trained_model_path)
                            # Update the MCTS player with the new network reference
                            mcts_player.network = chess_network
                            print("Successfully reloaded tactics-trained model into the current session.")
                        else:
                            print(f"[ERROR] Expected tactics-trained model not found at {tactics_trained_model_path}. Continuing with the old model.")
                    
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] The tactics training script failed with exit code {e.returncode}. Continuing with the old model.")
                    except Exception as e:
                        print(f"[ERROR] An unexpected error occurred during tactics training: {e}. Continuing with the old model.")
            
            print("\n" + "="*80)
            print("--- TACTICS SESSION COMPLETE ---")
            print("="*80 + "\n")


        previous_mode = current_mode
        
        # --- FINAL LOGIC: Implement Grace Period ---
        # Determine how many games have passed since the last mentor session.
        log_df = pd.read_csv(loss_log_filepath) if os.path.exists(loss_log_filepath) else pd.DataFrame(columns=['game', 'game_type'])
        mentor_games = log_df[log_df['game_type'] == 'mentor-play']
        last_mentor_game = mentor_games['game'].max() if not mentor_games.empty else -1
        
        games_since_mentor = game_num - last_mentor_game
        
        # If we are within the grace period, force self-play.
        if last_mentor_game != -1 and games_since_mentor <= config_params['SUPERVISOR_GRACE_PERIOD']:
            print(f"Within grace period ({games_since_mentor}/{config_params['SUPERVISOR_GRACE_PERIOD']} games since mentor). Forcing self-play.")
            current_mode = "self-play"
        else:
            # If outside the grace period, let the supervisor check for stagnation.
            stagnation_detected = supervisor.check_for_stagnation(loss_log_filepath)
            if stagnation_detected:
                current_mode = "mentor-play"
            else:
                current_mode = "self-play"
        
        # Log the mode switch if it occurred
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