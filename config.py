# FILENAME: config.py (Updated for Phase BI)
import os
from pathlib import Path
from collections import namedtuple

# =================================================================
# 1. Hyperparameter Configuration
# =================================================================

config_params = {
    # -- General & Path Settings --
    "DEVICE": "auto",
    "STOCKFISH_PATH": "/usr/games/stockfish",

    # -- Training Run Settings --
    "TOTAL_GAMES": 2000,
    "CHECKPOINT_INTERVAL": 10,
    "TRAINING_EPOCHS": 1,
    "BATCH_SIZE": 64,

    # -- MCTS Settings --
    "MCTS_SIMULATIONS": 800,
    "CPUCT": 2.5,
    
    # -- Tactical Puzzle Settings --
    "TACTICAL_PUZZLE_FILENAME": "tactical_puzzles.jsonl",
    "GENERATED_PUZZLE_FILENAME": "generated_puzzles.jsonl",
    "PUZZLE_RATIO": 0.25,

    # -- Phase AR: Tactical Primer Settings --
    "TACTICAL_PRIMER_BATCHES": 1, 
    
    # -- Supervisor Parameters --
    'SUPERVISOR_WINDOW_SIZE': 20,
    'SUPERVISOR_PERFORMANCE_THRESHOLD': 7.0,
    
    # -- Bayesian Supervisor Specific --
    'SUPERVISOR_BAYESIAN_PENALTY': 0.8,
    'SUPERVISOR_RECENCY_WINDOW': 50,
    'SUPERVISOR_GRACE_PERIOD': 10, 

    # -- Mentor & Opponent Settings --
    "MENTOR_ELO": 2000,
    "MENTOR_GAME_AGENT_COLOR": "random",
    "STOCKFISH_DEPTH_MENTOR": 10,
    "STOCKFISH_DEPTH_EVAL": 10,

    # -- Neural Network & Training Settings --
    "GNN_EMBED_DIM": 256,
    "CNN_EMBED_DIM": 256,
    "GNN_HIDDEN_DIM": 128,
    "NUM_HEADS": 4,
    "LEARNING_RATE": 0.0001,
    "WEIGHT_DECAY": 0.0001,
    "VALUE_LOSS_WEIGHT": 1.0,

    # -- LR Scheduler Settings (Phase AG) --
    'LR_SCHEDULER_STEP_SIZE': 100,
    'LR_SCHEDULER_GAMMA': 0.9,

    # --- PHASE BH: Added Contempt Factor to discourage draws ---
    "CONTEMPT_FACTOR": -0.1,
    
    # --- PHASE BI: Added Material Balance Loss Weight ---
    "MATERIAL_BALANCE_LOSS_WEIGHT": 0.5,
}


# =================================================================
# 2. Path Configuration (Colab-aware)
# =================================================================
# (The rest of this file is unchanged)

Paths = namedtuple('Paths', [
    'checkpoints_dir', 
    'training_data_dir', 
    'pgn_games_dir', 
    'analysis_output_dir',
    'tactical_puzzles_file',
    'generated_puzzles_file',
    'drive_project_root',
    'loss_log_file',
    'supervisor_log_file'
])

def get_paths():
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected. Using pre-mounted Google Drive paths.")
        drive_root_path = Path('/content/drive/MyDrive/ChessMCTS_RL')
        if not Path('/content/drive').is_dir():
            raise IOError(
                "Google Drive is not mounted. Please mount it in a Colab cell."
            )
    else:
        print("Running in a local environment.")
        drive_root_path = Path.cwd()
    
    checkpoints_path = drive_root_path / 'checkpoints'
    training_data_path = drive_root_path / 'training_data'
    pgn_games_path = drive_root_path / 'pgn_games'
    analysis_output_path = drive_root_path / 'analysis_output'

    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)
    pgn_games_path.mkdir(parents=True, exist_ok=True)
    analysis_output_path.mkdir(parents=True, exist_ok=True)
    
    tactical_puzzles_path = drive_root_path / config_params["TACTICAL_PUZZLE_FILENAME"]
    generated_puzzles_path = drive_root_path / config_params["GENERATED_PUZZLE_FILENAME"]
    loss_log_filepath = drive_root_path / 'loss_log_v2.csv'
    supervisor_log_filepath = drive_root_path / 'supervisor_log.txt'
    
    return Paths(
        checkpoints_dir=checkpoints_path,
        training_data_dir=training_data_path,
        pgn_games_dir=pgn_games_path,
        analysis_output_dir=analysis_output_path,
        tactical_puzzles_file=tactical_puzzles_path,
        generated_puzzles_file=generated_puzzles_path,
        drive_project_root=drive_root_path,
        loss_log_file=loss_log_filepath,
        supervisor_log_file=supervisor_log_filepath
    )
