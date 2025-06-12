import os
import torch
import pandas as pd
from pathlib import Path
import chess.pgn

# --- Import from config ---
from config import get_paths, config_params

# Core components from the gnn_agent package
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.search.mcts import MCTS
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.mentor_play import MentorPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer


def write_loss_to_csv(filepath, game_num, policy_loss, value_loss, game_type):
    """Appends a new row of loss data to a CSV file."""
    file_exists = os.path.isfile(filepath)
    df = pd.DataFrame([[game_num, policy_loss, value_loss, game_type]], columns=['game', 'policy_loss', 'value_loss', 'game_type'])
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)

def main():
    """
    Main training loop that orchestrates self-play, mentor-play, and network training.
    """
    # --- 1. Get Environment-Aware Paths & Config ---
    checkpoints_path, training_data_path = get_paths()
    
    # --- ADDED: Create and print a path for PGNs ---
    project_root = training_data_path.parent
    pgn_path = project_root / 'pgn_games'
    pgn_path.mkdir(exist_ok=True)
    
    loss_log_filepath = project_root / 'loss_log_v2.csv'

    device_str = config_params['DEVICE']
    device = "cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu"

    print(f"Using device: {device}")
    print(f"Checkpoints will be saved to: {checkpoints_path}")
    print(f"Training data will be saved to: {training_data_path}")
    print(f"PGN games will be saved to: {pgn_path}")
    print(f"Losses will be logged to: {loss_log_filepath}")

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
    
    # --- 4. Initialize Players ---
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

    # --- 5. Main Training Loop ---
    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        
        training_examples = []
        pgn_data = None
        game_type = ""

        # a. Alternate between Mentor and Self-Play Games
        if game_num % config_params['MENTOR_GAME_INTERVAL'] == 0:
            game_type = "mentor"
            print(f"\n--- Starting Mentor Game {game_num}/{config_params['TOTAL_GAMES']} ---")
            # --- UPDATED: Handle two return values ---
            training_examples, pgn_data = mentor_player.play_game()
        else:
            game_type = "self-play"
            print(f"\n--- Starting Self-Play Game {game_num}/{config_params['TOTAL_GAMES']} ---")
            # --- UPDATED: Handle two return values ---
            training_examples, pgn_data = self_player.play_game()

        if not training_examples:
            print(f"Game type '{game_type}' resulted in no training examples. Skipping.")
            continue
        print(f"{game_type.capitalize()} game complete. Generated {len(training_examples)} examples.")

        # b. Save the new data and PGN
        data_filename = f"{game_type}_game_{game_num}_data.pkl"
        training_data_manager.save_data(training_examples, filename=data_filename)
        
        # --- ADDED: Save PGN to file ---
        if pgn_data:
            pgn_filename = pgn_path / f"{game_type}_game_{game_num}.pgn"
            try:
                with open(pgn_filename, "w", encoding="utf-8") as f:
                    print(pgn_data, file=f, end="\n\n")
                print(f"Successfully saved PGN to {pgn_filename}")
            except Exception as e:
                print(f"[ERROR] Could not save PGN file: {e}")

        # c. Train the network
        print(f"Training on the {len(training_examples)} examples from game {game_num}...")
        for epoch in range(config_params['TRAINING_EPOCHS']):
            policy_loss, value_loss = trainer.train_on_batch(training_examples, batch_size=config_params['BATCH_SIZE'])
            print(f"Epoch {epoch + 1}/{config_params['TRAINING_EPOCHS']} complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            write_loss_to_csv(loss_log_filepath, game_num, policy_loss, value_loss, game_type)

        # d. Save a checkpoint periodically
        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=checkpoints_path, game_number=game_num)

    print("\n--- Training Run Finished ---")
    self_player.close()
    mentor_player.close()


if __name__ == "__main__":
    main()