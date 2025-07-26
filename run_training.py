# FILENAME: run_training.py

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

# --- MODIFIED: Import the new model ---
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from hardware_setup import get_device, install_xla_if_needed
from gnn_agent.search.mcts import MCTS
from gnn_agent.rl_loop.self_play import SelfPlay
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager
from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.rl_loop.bayesian_supervisor import BayesianSupervisor
from gnn_agent.rl_loop.guided_session import run_guided_session

# --- MODIFIED: Added for initializing pre-trained model ---
# This metadata is required to initialize the UnifiedGNN within your model.
GNN_METADATA = (
    ['square', 'piece'],
    [
        ('square', 'adjacent_to', 'square'),
        ('piece', 'occupies', 'square'),
        ('piece', 'attacks', 'piece'),
        ('piece', 'defends', 'piece')
    ]
)

class MoveHandler:
    """
    A self-contained class to handle move conversions for model initialization.
    """
    def __init__(self):
        self.move_map = self._create_move_map()
        self.policy_size = len(self.move_map)

    def _create_move_map(self):
        moves = {}
        idx = 0
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq == to_sq:
                    continue
                for prom in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves[chess.Move(from_sq, to_sq, promotion=prom)] = idx
                    idx += 1
                moves[chess.Move(from_sq, to_sq)] = idx
                idx += 1
        return moves
# --- End of added block ---

def write_loss_to_csv(filepath, game_num, policy_loss, value_loss, next_state_loss, game_type):
    file_exists = os.path.isfile(filepath)
    df = pd.DataFrame([[game_num, policy_loss, value_loss, next_state_loss, game_type]],
                      columns=['game', 'policy_loss', 'value_loss', 'next_state_loss', 'game_type'])
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
    # --- MODIFIED: Added new argument for loading a pre-trained model ---
    parser.add_argument('--load-pretrained-checkpoint', type=str, default=None, help="Path to a pre-trained model checkpoint to start the run.")
    args = parser.parse_args()

    if args.disable_puzzle_mixing:
        print("\n" + "#"*60 + "\n--- TACTICAL PUZZLE MIXING IS DISABLED FOR THIS RUN ---\n" + "#"*60 + "\n")

    paths = get_paths()
    
    device = get_device()
    install_xla_if_needed(device)
    print(f"Using device: {device}")

    trainer = Trainer(model_config=config_params, device=device)
    
    # --- MODIFIED: Logic to handle loading pre-trained model vs. resuming ---
    if args.load_pretrained_checkpoint:
        print("\n" + "#"*60)
        print("--- LOADING PRE-TRAINED MODEL ---")
        pretrained_path = Path(args.load_pretrained_checkpoint)
        if not pretrained_path.exists():
            print(f"[FATAL] Pre-trained checkpoint not found at: {pretrained_path}")
            sys.exit(1)
            
        move_handler = MoveHandler()
        chess_network = ValueNextStateModel(
            gnn_hidden_dim=config_params['GNN_HIDDEN_DIM'],
            cnn_in_channels=14, 
            embed_dim=config_params['EMBED_DIM'],
            policy_size=move_handler.policy_size,
            gnn_num_heads=config_params['GNN_NUM_HEADS'],
            gnn_metadata=GNN_METADATA
        ).to(device)
        
        chess_network.load_state_dict(torch.load(pretrained_path, map_location=device))
        start_game = 0
        trainer.network = chess_network # Ensure the trainer has the loaded network
        print(f"Successfully loaded pre-trained model from: {pretrained_path}")
        print("Starting new training run from Game 1.")
        print("#"*60 + "\n")
    else:
        chess_network, start_game = trainer.load_or_initialize_network(directory=paths.checkpoints_dir)
        print(f"Resuming training run from game {start_game + 1}")
    # --- END OF MODIFIED BLOCK ---

    try:
        print("\n" + "-"*45)
        print("--- Model Architecture Verification ---")
        if isinstance(chess_network, ValueNextStateModel):
            print("   - ValueNextStateModel (GNN+CNN) Detected.")
        else:
            print("   - [WARNING] Could not verify model structure.")
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
        training_examples = []
        pgn_data = None
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
                        primer_policy_loss, _, _ = trainer.train_on_batch(game_examples=[], puzzle_examples=puzzles_for_primer, batch_size=config_params['BATCH_SIZE'])
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
        else: 
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

        print("\n" + "="*20 + " DEBUG INFO " + "="*20)
        print(f"Current Mode: {current_mode}")
        print(f"Data going to trainer - Game Examples: {len(game_examples_for_trainer)} games")
        print(f"Data going to trainer - Puzzle Examples: {len(puzzles_for_training)} puzzles")

        policy_loss, value_loss, next_state_loss = trainer.train_on_batch(
            game_examples=game_examples_for_trainer,
            puzzle_examples=puzzles_for_training,
            batch_size=config_params['BATCH_SIZE']
        )
        print(f"Training complete. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Next-State Loss: {next_state_loss:.4f}")
        
        write_loss_to_csv(paths.loss_log_file, game_num, policy_loss, value_loss, next_state_loss, current_mode)

        if game_num % config_params['CHECKPOINT_INTERVAL'] == 0:
            print(f"Saving checkpoint at game {game_num}...")
            trainer.save_checkpoint(directory=paths.checkpoints_dir, game_number=game_num)
            
    print("\n--- Training Run Finished ---")
    self_player.close()
    mentor_engine.quit()

if __name__ == "__main__":
    main()