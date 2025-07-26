# FILENAME: run_mentor_play.py

import os
import torch
import torch.optim as optim
import pandas as pd
from pathlib import Path
import chess
import chess.pgn
import datetime
import sys
import argparse
import random
from stockfish import Stockfish
from typing import List, Tuple, Dict, Any

# --- Import from config ---
from config import get_paths, config_params

# --- Project-specific Imports ---
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from hardware_setup import get_device, install_xla_if_needed
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.gamestate_converters.action_space_converter import move_to_index, get_action_space_size

# --- GNN Metadata (copied from run_training.py for self-containment) ---
GNN_METADATA = (
    ['square', 'piece'],
    [
        ('square', 'adjacent_to', 'square'),
        ('piece', 'occupies', 'square'),
        ('piece', 'attacks', 'piece'),
        ('piece', 'defends', 'piece')
    ]
)

def write_loss_to_csv(filepath: Path, game_num: int, policy_loss: float, value_loss: float, next_state_loss: float):
    """Appends the loss values for a completed training game to a CSV log."""
    file_exists = filepath.is_file()
    df = pd.DataFrame([[game_num, policy_loss, value_loss, next_state_loss, 'mentor-play']],
                      columns=['game', 'policy_loss', 'value_loss', 'next_state_loss', 'game_type'])
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)


def get_mentor_evaluation(mentor_engine: Stockfish, board: chess.Board) -> Tuple[chess.Move, float]:
    """Gets the best move and scaled evaluation from the mentor engine."""
    mentor_engine.set_fen_position(board.fen())
    best_move_uci = mentor_engine.get_best_move_time(100) # 100ms per move
    if not best_move_uci:
        return None, 0.0
    
    move = chess.Move.from_uci(best_move_uci)
    
    eval_result = mentor_engine.get_evaluation()
    if eval_result['type'] == 'mate':
        score = 30000 if eval_result['value'] > 0 else -30000
    else:
        score = eval_result['value'] # Centipawns

    # Convert centipawns to a value between -1 and 1
    # We use tanh to squash the value, scaled by a factor to make it sensitive in the typical chess evaluation range.
    scaled_value = torch.tanh(torch.tensor(score / 1000.0)).item()
    
    # The returned value is from the perspective of the current player
    return move, scaled_value


def play_one_mentor_game(mentor_engine: Stockfish, contempt_factor: float) -> Tuple[List[Tuple[str, Dict, float, float]], chess.pgn.Game]:
    """
    Plays one full game where the mentor engine chooses all moves.
    Generates training data (FEN, policy, value, next_state_value) for each move.
    """
    board = chess.Board()
    game_history = [] # Stores (fen, move, value_for_state) tuples
    
    while not board.is_game_over(claim_draw=True):
        current_fen = board.fen()
        turn_before_move = board.turn
        
        move, value = get_mentor_evaluation(mentor_engine, board)
        
        if move is None:
            break

        game_history.append((current_fen, move, value, turn_before_move))
        board.push(move)

    # --- Game is Over: Assign final outcomes and next-state values ---
    training_examples = []
    for i, (fen, move, value, turn) in enumerate(game_history):
        # The policy is a sparse dictionary with 1.0 probability for the mentor's move
        policy_target = {move: 1.0}
        
        # The value is the mentor's evaluation for the current state
        value_target = value

        # The next_state_value is the mentor's evaluation of the *next* state, from the current player's perspective
        if i + 1 < len(game_history):
            next_fen, _, next_value, next_turn = game_history[i+1]
            # next_value is from the perspective of the *next* player. We need to flip it if the player changes.
            next_state_value_target = next_value if turn == next_turn else -next_value
        else:
            # For the last move, the next state is terminal.
            result = board.result(claim_draw=True)
            if result == '1-0':
                next_state_value_target = 1.0 if turn == chess.WHITE else -1.0
            elif result == '0-1':
                next_state_value_target = -1.0 if turn == chess.WHITE else 1.0
            else: # Draw
                next_state_value_target = contempt_factor

        training_examples.append((fen, policy_target, value_target, next_state_value_target))

    # --- Generate PGN ---
    pgn = chess.pgn.Game.from_board(board)
    pgn.headers["Event"] = "Mentor-Play Training Game"
    pgn.headers["Site"] = "Juprelle, Wallonia, Belgium"
    pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn.headers["White"] = "Mentor (Stockfish)"
    pgn.headers["Black"] = "Mentor (Stockfish)"
    pgn.headers["Result"] = board.result(claim_draw=True)
    
    return training_examples, pgn


def main():
    parser = argparse.ArgumentParser(description="Run the Mentor-Play RL training loop.")
    parser.add_argument('--load-pretrained-checkpoint', type=str, required=True, help="Path to a pre-trained model checkpoint to start the run.")
    parser.add_argument('--num-games', type=int, default=config_params['TOTAL_GAMES'], help="Number of mentor games to play and train on.")
    args = parser.parse_args()

    paths = get_paths()
    device = get_device()
    install_xla_if_needed(device)
    print(f"Using device: {device}")

    # --- Initialize Trainer ---
    # The trainer class will hold the network and optimizer
    trainer = Trainer(model_config=config_params, device=device)
    
    # --- Load the Pre-trained Model ---
    print("\n" + "#"*60)
    print("--- LOADING PRE-TRAINED MODEL FOR MENTOR-PLAY ---")
    pretrained_path = Path(args.load_pretrained_checkpoint)
    if not pretrained_path.exists():
        print(f"[FATAL] Pre-trained checkpoint not found at: {pretrained_path}")
        sys.exit(1)
        
    # Instantiate the network architecture
    chess_network = ValueNextStateModel(
        gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'],
        cnn_in_channels=14, 
        embed_dim=config_params['EMBED_DIM'],
        policy_size=get_action_space_size(),
        gnn_num_heads=config_params['GNN_NUM_HEADS'],
        gnn_metadata=GNN_METADATA
    ).to(device)
    
    # Load the learned weights
    chess_network.load_state_dict(torch.load(pretrained_path, map_location=device))
    trainer.network = chess_network 
    
    # Initialize the optimizer for the trainer
    trainer.optimizer = optim.AdamW(
        chess_network.parameters(),
        lr=config_params['LEARNING_RATE'],
        weight_decay=config_params['WEIGHT_DECAY']
    )
    
    print(f"Successfully loaded pre-trained model from: {pretrained_path}")
    print("#"*60 + "\n")

    # --- Initialize Mentor Engine ---
    try:
        mentor_engine = Stockfish(path=config_params['STOCKFISH_PATH'], depth=config_params['STOCKFISH_DEPTH_MENTOR'])
        mentor_engine.set_elo_rating(config_params['MENTOR_ELO'])
    except Exception as e:
        print(f"[FATAL] Could not initialize the Stockfish engine: {e}\n" + "Please ensure the STOCKFISH_PATH in config.py is correct and the binary is executable.")
        sys.exit(1)

    training_data_manager = TrainingDataManager(data_directory=paths.training_data_dir)
        
    # --- Main Mentor-Play Loop ---
    for game_num in range(1, args.num_games + 1):
        print(f"\n--- Game {game_num}/{args.num_games} (Mode: MENTOR-PLAY) ---")
        
        # 1. Generate data by playing a game with the mentor
        training_examples, pgn_data = play_one_mentor_game(mentor_engine, config_params.get('CONTEMPT_FACTOR', 0.0))

        if not training_examples:
            print("Mentor game resulted in no examples. Skipping cycle.")
            continue
        
        print(f"Mentor game complete. Generated {len(training_examples)} examples.")
        
        # 2. Save the generated data and PGN
        data_filename = f"mentor-play_game_{game_num}_data.pkl"
        training_data_manager.save_data(training_examples, filename=data_filename)
        
        pgn_filename = paths.pgn_games_dir / f"mentor-play_game_{game_num}.pgn"
        with open(pgn_filename, "w", encoding="utf-8") as f:
            print(pgn_data, file=f, end="\n\n")

        # 3. Train the network on the new data
        # The trainer expects a list of games, so we wrap our examples in a list
        policy_loss, value_loss, next_state_loss = trainer.train_on_batch(
            game_examples=[training_examples], 
            puzzle_examples=[], # No puzzles in this mode
            batch_size=config_params['BATCH_SIZE']
        )
        print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Next-State Loss: {next_state_loss:.4f}")
        
        # 4. Log the loss
        write_loss_to_csv(paths.loss_log_file, game_num, policy_loss, value_loss, next_state_loss)

        # 5. Save a checkpoint periodically
        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=paths.checkpoints_dir, game_number=game_num)
            
    print("\n--- Mentor-Play Training Run Finished ---")
    mentor_engine.quit()

if __name__ == "__main__":
    main()