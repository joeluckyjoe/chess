import os
import torch
import pandas as pd
from pathlib import Path
import chess.pgn
import datetime

# --- Import from config ---
from config import get_paths, config_params

# Core components from the gnn_agent package
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.search.mcts import MCTS
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.mentor_play import MentorPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.rl_loop.threshold_supervisor import ThresholdSupervisor


def write_loss_to_csv(filepath, game_num, policy_loss, value_loss, game_type):
    """Appends a new row of loss data to a CSV file."""
    file_exists = os.path.isfile(filepath)
    df = pd.DataFrame([[game_num, policy_loss, value_loss, game_type]], columns=['game', 'policy_loss', 'value_loss', 'game_type'])
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)

def main():
    """
    Main training loop that orchestrates self-play, mentor-play, and network training,
    guided by the ThresholdSupervisor.
    """
    # --- 1. Get Environment-Aware Paths & Config ---
    checkpoints_path, training_data_path = get_paths()
    
    # --- MODIFIED: Use the correct base path for logs ---
    # In Colab, training_data_path will be on Google Drive. Its parent is the project root.
    # In local, it's the local project root. This ensures logs are saved with the data.
    project_root = training_data_path.parent 
    pgn_path = project_root / 'pgn_games'
    pgn_path.mkdir(parents=True, exist_ok=True)
    
    loss_log_filepath = project_root / 'loss_log_v2.csv'
    supervisor_log_filepath = project_root / 'supervisor_log.txt'

    device_str = config_params['DEVICE']
    device = "cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu"

    print(f"Using device: {device}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")
    print(f"Training data (and logs) will be saved to: {project_root}")

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

    print("Initializing Threshold Supervisor...")
    supervisor = ThresholdSupervisor(config=config_params)

    # --- 5. Main Training Loop ---
    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        
        current_mode = supervisor.mode
        print(f"\n--- Game {game_num}/{config_params['TOTAL_GAMES']} (Current Mode: {current_mode}) ---")
        
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

        previous_mode = supervisor.mode
        supervisor.update({
            'policy_loss': final_policy_loss,
            'value_loss': final_value_loss,
            'num_moves': num_moves
        })
        
        if supervisor.mode != previous_mode:
            reason_for_switch = f"Supervisor switched mode from '{previous_mode}' to '{supervisor.mode}'."
            with open(supervisor_log_filepath, 'a') as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{timestamp}] Game {game_num}: {reason_for_switch}\n"
                f.write(log_message)

        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=checkpoints_path, game_number=game_num)

    print("\n--- Training Run Finished ---")
    self_player.close()
    mentor_player.close()

if __name__ == "__main__":
    main()
