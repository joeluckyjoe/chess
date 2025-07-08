# run_training.py

import os
import torch
import pandas as pd
from pathlib import Path
import chess.pgn
import datetime
import random
import sys
import argparse
import json
from stockfish import Stockfish

# --- Import from config ---
from config import get_paths, config_params

# Core components from the gnn_agent package
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.search.mcts import MCTS
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.rl_loop.bayesian_supervisor import BayesianSupervisor
from gnn_agent.rl_loop.guided_session import run_guided_session # <-- NEW IMPORT

# (The functions write_loss_to_csv, is_in_grace_period, load_puzzles_from_sources remain unchanged)

def write_loss_to_csv(filepath, game_num, policy_loss, value_loss, game_type):
    """Appends a new row of loss data to a CSV file."""
    file_exists = os.path.isfile(filepath)
    df = pd.DataFrame([[game_num, policy_loss, value_loss, game_type]], columns=['game', 'policy_loss', 'value_loss', 'game_type'])
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)

def is_in_grace_period(log_file_path: Path, grace_period_length: int) -> bool:
    """
    Checks if a mentor game has occurred within the last N games,
    indicating that the system is in a grace period.
    """
    if not log_file_path.exists():
        return False
    try:
        df = pd.read_csv(log_file_path)
        if df.empty or 'game_type' not in df.columns:
            return False
        # Check the 'game_type' in the last N rows for any intervention type
        recent_games = df.tail(grace_period_length)
        intervention_types = ['mentor-play', 'guided-mentor-session']
        return any(game_type in recent_games['game_type'].values for game_type in intervention_types)
    except Exception as e:
        print(f"[WARNING] Could not check grace period due to error: {e}")
        return False

def load_puzzles_from_sources(puzzle_paths: list[Path]):
    """
    Loads tactical puzzles from a list of .jsonl file paths,
    combining them into a single list.
    """
    all_puzzles = []
    print("Loading tactical puzzles...")
    for path in puzzle_paths:
        if not path.exists():
            print(f"   - INFO: Puzzle file not found at {path}. Skipping.")
            continue
        try:
            with open(path, 'r') as f:
                puzzles_from_file = [json.loads(line) for line in f]
                all_puzzles.extend(puzzles_from_file)
                print(f"   - Successfully loaded {len(puzzles_from_file)} puzzles from {os.path.basename(str(path))}.")
        except Exception as e:
            print(f"   - ERROR: Failed to load puzzles from {path}: {e}")
            
    if not all_puzzles:
        print("[WARNING] No tactical puzzles were loaded in total. Interventions may fail.")
    else:
        print(f"Total puzzles loaded: {len(all_puzzles)}")
        
    return all_puzzles


def main():
    """
    Main training loop that orchestrates self-play, mentor-play, and network training,
    now featuring the two-stage "Tactical Primer" intervention protocol.
    """
    parser = argparse.ArgumentParser(description="Run the MCTS RL training loop.")
    parser.add_argument(
        '--disable-puzzle-mixing',
        action='store_true',
        help="If set, disables the mixing of tactical puzzles during standard training."
    )
    args = parser.parse_args()

    if args.disable_puzzle_mixing:
        print("\n" + "#"*60)
        print("--- TACTICAL PUZZLE MIXING IS DISABLED FOR THIS RUN ---")
        print("#"*60 + "\n")

    paths = get_paths()
    device_str = config_params['DEVICE']
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu")
    print(f"Using device: {device}")

    # --- Initialization ---
    trainer = Trainer(model_config=config_params, device=device)
    
    chess_network, start_game = trainer.load_or_initialize_network(directory=paths.checkpoints_dir)
    
    print(f"Resuming training run from game {start_game + 1}")

    all_puzzle_sources = [paths.tactical_puzzles_file, paths.generated_puzzles_file]
    all_puzzles = load_puzzles_from_sources(all_puzzle_sources)

    mcts_player = MCTS(
        network=chess_network, device=device,
        c_puct=config_params['CPUCT'], batch_size=config_params['BATCH_SIZE']
    )
    
    self_player = SelfPlay(
        mcts_player=mcts_player, num_simulations=config_params['MCTS_SIMULATIONS']
    )

    # NEW: Instantiate a separate Stockfish engine for the guided mentor session
    try:
        mentor_engine = Stockfish(path=config_params['STOCKFISH_PATH'], depth=15)
        mentor_engine.set_elo_rating(config_params['MENTOR_ELO'])
    except Exception as e:
        print(f"[FATAL] Could not initialize the Stockfish engine: {e}")
        print("Please ensure the STOCKFISH_PATH in config.py is correct and the binary is executable.")
        sys.exit(1)

    training_data_manager = TrainingDataManager(data_directory=paths.training_data_dir)
    supervisor = BayesianSupervisor(config=config_params)
        
    # --- Main Training Loop ---
    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        
        current_mode = "self-play" # Default to self-play
        training_examples, pgn_data = [], None # Initialize loop variables

        grace_period_active = is_in_grace_period(
            paths.loss_log_file,
            config_params.get('SUPERVISOR_GRACE_PERIOD', 10)
        )

        if grace_period_active:
            print(f"\nINFO: Post-intervention grace period active for Game {game_num}. Forcing self-play.")
            # Mode is already self-play
        else:
            print("\nINFO: Consulting Bayesian Supervisor for stagnation check...")
            needs_intervention = supervisor.check_for_stagnation(paths.loss_log_file)
            
            if needs_intervention:
                print("\n" + "="*70)
                print(f"STAGNATION DETECTED: Initiating GUIDED MENTOR SESSION for Game {game_num}.")
                print("="*70)

                # --- Stage 1: Tactical Primer ---
                print("\n--- Stage 1: Tactical Primer ---")
                if all_puzzles:
                    num_primer_batches = config_params.get('TACTICAL_PRIMER_BATCHES', 1)
                    primer_batch_size = config_params['BATCH_SIZE']
                    num_puzzles_for_primer = num_primer_batches * primer_batch_size
                    
                    if len(all_puzzles) >= num_puzzles_for_primer:
                        puzzles_for_primer = random.sample(all_puzzles, num_puzzles_for_primer)
                        print(f"Executing tactical primer with {len(puzzles_for_primer)} puzzles.")
                        primer_policy_loss, _ = trainer.train_on_batch(
                            game_examples=[], puzzle_examples=puzzles_for_primer,
                            batch_size=primer_batch_size, puzzle_ratio=1.0
                        )
                        print(f"Tactical Primer Complete. Policy Loss: {primer_policy_loss:.4f}")
                    else:
                        print(f"[WARNING] Not enough puzzles ({len(all_puzzles)}) for a full primer. Skipping.")
                else:
                    print("[WARNING] No tactical puzzles loaded. Cannot execute tactical primer.")

                # --- Stage 2: NEW Guided Mentor Session ---
                print("\n--- Stage 2: Guided Mentor Session ---")
                current_mode = "guided-mentor-session"
                
                # Directly call the new guided session function
                training_examples, pgn_data = run_guided_session(
                    agent=chess_network,
                    mentor_engine=mentor_engine,
                    search_manager=mcts_player,
                    value_threshold=config_params.get('GUIDED_SESSION_VALUE_THRESHOLD', 0.1),
                    agent_color_str=config_params['MENTOR_GAME_AGENT_COLOR']
                )
        
        # --- Play Game (if not already played in an intervention) ---
        if not training_examples:
             print(f"\n--- Game {game_num}/{config_params['TOTAL_GAMES']} (Mode: {current_mode.upper()}) ---")
             training_examples, pgn_data = self_player.play_game()

        # --- Process Game Data ---
        if not training_examples:
            print(f"Game type '{current_mode}' resulted in no examples. Skipping training and saving for this cycle.")
            continue
        
        print(f"{current_mode.capitalize()} game complete. Generated {len(training_examples)} examples.")
        
        data_filename = f"{current_mode}_game_{game_num}_data.pkl"
        training_data_manager.save_data(training_examples, filename=data_filename)
        
        if pgn_data:
            pgn_filename = paths.pgn_games_dir / f"{current_mode}_game_{game_num}.pgn"
            try:
                with open(pgn_filename, "w", encoding="utf-8") as f:
                    print(pgn_data, file=f, end="\n\n")
            except Exception as e:
                print(f"[ERROR] Could not save PGN file: {e}")

        # --- Standard Training Step on New Data ---
        puzzles_for_training = [] if args.disable_puzzle_mixing else all_puzzles
        
        print(f"Training on {len(training_examples)} new examples...")
        policy_loss, value_loss = trainer.train_on_batch(
            game_examples=training_examples, puzzle_examples=puzzles_for_training,
            batch_size=config_params['BATCH_SIZE'],
            puzzle_ratio=config_params.get('PUZZLE_RATIO', 0.25)
        )
        
        print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        write_loss_to_csv(paths.loss_log_file, game_num, policy_loss, value_loss, current_mode)

        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=paths.checkpoints_dir, game_number=game_num)
            
    print("\n--- Training Run Finished ---")
    self_player.close()
    mentor_engine.quit() # Close the new mentor engine instance

if __name__ == "__main__":
    main()