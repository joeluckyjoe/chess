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

from gnn_agent.neural_network.hybrid_transformer_model import HybridTransformerModel
from hardware_setup import get_device, install_xla_if_needed
from gnn_agent.search.mcts import MCTS
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.rl_loop.bayesian_supervisor import BayesianSupervisor
from gnn_agent.rl_loop.guided_session import run_guided_session

def write_loss_to_csv(filepath, game_num, policy_loss, value_loss, material_loss, game_type):
    file_exists = os.path.isfile(filepath)
    df = pd.DataFrame([[game_num, policy_loss, value_loss, material_loss, game_type]],
                      columns=['game', 'policy_loss', 'value_loss', 'material_loss', 'game_type'])
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)

def is_in_grace_period(log_file_path: Path, grace_period_length: int) -> bool:
    if not log_file_path.exists(): return False
    try:
        df = pd.read_csv(log_file_path)
        if df.empty or 'game_type' not in df.columns: return False
        recent_games = df.tail(grace_period_length)
        intervention_types = ['mentor-play', 'guided-mentor-session']
        return any(game_type in recent_games['game_type'].values for game_type in intervention_types)
    except Exception as e:
        print(f"[WARNING] Could not check grace period due to error: {e}")
        return False

def load_puzzles_from_sources(puzzle_paths: list[Path]):
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
    parser = argparse.ArgumentParser(description="Run the MCTS RL training loop.")
    parser.add_argument('--disable-puzzle-mixing', action='store_true', help="If set, disables the mixing of tactical puzzles during standard training.")
    args = parser.parse_args()

    if args.disable_puzzle_mixing:
        print("\n" + "#"*60 + "\n--- TACTICAL PUZZLE MIXING IS DISABLED FOR THIS RUN ---\n" + "#"*60 + "\n")

    paths = get_paths()
    
    device = get_device()
    install_xla_if_needed(device)
    print(f"Using device: {device}")

    trainer = Trainer(model_config=config_params, device=device)
    chess_network, start_game = trainer.load_or_initialize_network(directory=paths.checkpoints_dir)
    print(f"Resuming training run from game {start_game + 1}")

    try:
        print("\n" + "-"*45)
        print("--- Model Architecture Verification ---")
        if isinstance(chess_network, HybridTransformerModel):
             print("   - GNN+CNN+Transformer Hybrid Model Detected.")
        else:
             print("   - [WARNING] Could not verify hybrid structure.")
        print("-"*45 + "\n")
    except Exception as e:
        print(f"\n[WARNING] An unexpected error occurred during feature verification: {e}\n")

    all_puzzles = load_puzzles_from_sources([paths.tactical_puzzles_file, paths.generated_puzzles_file])

    mcts_player = MCTS(network=chess_network, device=device, c_puct=config_params['CPUCT'], batch_size=config_params['BATCH_SIZE'])
    
    self_player = SelfPlay(
        network=chess_network,
        device=device,
        mcts_white=mcts_player, 
        mcts_black=mcts_player, 
        stockfish_path=config_params['STOCKFISH_PATH'], 
        num_simulations=config_params['MCTS_SIMULATIONS'],
        contempt_factor=config_params.get('CONTEMPT_FACTOR', 0.0)
    )

    try:
        mentor_engine = Stockfish(path=config_params['STOCKFISH_PATH'], depth=15)
        mentor_engine.set_elo_rating(config_params['MENTOR_ELO'])
    except Exception as e:
        print(f"[FATAL] Could not initialize the Stockfish engine: {e}\n" + "Please ensure the STOCKFISH_PATH in config.py is correct and the binary is executable.")
        sys.exit(1)

    training_data_manager = TrainingDataManager(data_directory=paths.training_data_dir)
    supervisor = BayesianSupervisor(config=config_params)
        
    for game_num in range(start_game + 1, config_params['TOTAL_GAMES'] + 1):
        
        current_mode = "self-play"
        training_examples, pgn_data = [], None
        puzzles_for_training = []

        if is_in_grace_period(paths.loss_log_file, config_params.get('SUPERVISOR_GRACE_PERIOD', 10)):
            print(f"\nINFO: Post-intervention grace period active for Game {game_num}. Forcing self-play.")
        else:
            print("\nINFO: Consulting Bayesian Supervisor for stagnation check...")
            if supervisor.check_for_stagnation(paths.loss_log_file):
                print("\n" + "="*70 + f"\nSTAGNATION DETECTED: Initiating GUIDED MENTOR SESSION for Game {game_num}.\n" + "="*70)
                current_mode = "guided-mentor-session"

                print("\n--- Stage 1: Tactical Primer ---")
                if all_puzzles:
                    num_puzzles_for_primer = config_params.get('TACTICAL_PRIMER_BATCHES', 1) * config_params['BATCH_SIZE']
                    if len(all_puzzles) >= num_puzzles_for_primer:
                        puzzles_for_primer = random.sample(all_puzzles, num_puzzles_for_primer)
                        print(f"Executing tactical primer with {len(puzzles_for_primer)} puzzles.")
                        trainer.network = chess_network
                        primer_policy_loss, _, _ = trainer.train_on_batch(game_examples=[], puzzle_examples=puzzles_for_primer, batch_size=config_params['BATCH_SIZE'], puzzle_ratio=1.0)
                        print(f"Tactical Primer Complete. Policy Loss: {primer_policy_loss:.4f}")
                    else:
                        print(f"[WARNING] Not enough puzzles ({len(all_puzzles)}) for a full primer. Skipping.")
                else:
                    print("[WARNING] No tactical puzzles loaded. Cannot execute tactical primer.")

                print("\n--- Stage 2: Guided Mentor Session ---")
                training_examples, pgn_data = run_guided_session(
                    agent=chess_network,
                    mentor_engine=mentor_engine,
                    search_manager=mcts_player,
                    num_simulations=config_params['MCTS_SIMULATIONS'],
                    value_threshold=config_params.get('GUIDED_SESSION_VALUE_THRESHOLD', 0.1),
                    agent_color_str=config_params['MENTOR_GAME_AGENT_COLOR'],
                    contempt_factor=config_params.get('CONTEMPT_FACTOR', 0.0)
                )
        
        game_examples_for_trainer = []
        if current_mode == "self-play":
            print(f"\n--- Game {game_num}/{config_params['TOTAL_GAMES']} (Mode: {current_mode.upper()}) ---")
            training_examples, pgn_data = self_player.play_game()
            game_examples_for_trainer.append(training_examples)
            if not args.disable_puzzle_mixing:
                puzzles_for_training = all_puzzles
        else: # Guided session
            game_examples_for_trainer.append(training_examples)


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

        print(f"Training on {len(training_examples)} new examples...")
        trainer.network = chess_network
        
        # --- START DEBUG PRINTS ---
        print("\n" + "="*20 + " DEBUG INFO " + "="*20)
        print(f"Current Mode: {current_mode}")
        print(f"Type of game_examples_for_trainer: {type(game_examples_for_trainer)}")
        print(f"Is game_examples_for_trainer a list of lists?: {isinstance(game_examples_for_trainer[0], list)}")
        print(f"Number of games in batch: {len(game_examples_for_trainer)}")
        print(f"Number of training examples in first game: {len(game_examples_for_trainer[0])}")
        print(f"Type of puzzles_for_training: {type(puzzles_for_training)}")
        print(f"Number of puzzles for training: {len(puzzles_for_training)}")
        if puzzles_for_training:
            print(f"First puzzle sample: {puzzles_for_training[0]}")
        print("="*52 + "\n")
        # --- END DEBUG PRINTS ---

        policy_loss, value_loss, material_loss = trainer.train_on_batch(
            game_examples=game_examples_for_trainer,
            puzzle_examples=puzzles_for_training,
            batch_size=config_params['BATCH_SIZE'],
            puzzle_ratio=config_params.get('PUZZLE_RATIO', 0.25)
        )
        print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Material Loss: {material_loss:.4f}")
        
        write_loss_to_csv(paths.loss_log_file, game_num, policy_loss, value_loss, material_loss, current_mode)

        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=paths.checkpoints_dir, game_number=game_num)
            
    print("\n--- Training Run Finished ---")
    self_player.close()
    mentor_engine.quit()

if __name__ == "__main__":
    main()