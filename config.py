# FILENAME: config.py
import os
from pathlib import Path
from collections import namedtuple

# =================================================================
# 1. Hyperparameter Configuration
# =================================================================

config_params = {
    # -- General & Path Settings --
    "DEVICE": "auto",  # Use "auto" to detect CUDA, or force "cpu"
    "STOCKFISH_PATH": "/usr/games/stockfish",

    # -- Training Run Settings --
    "TOTAL_GAMES": 2000,        # Total games to run in the training session
    "CHECKPOINT_INTERVAL": 10,    # Save a checkpoint every N games
    "TRAINING_EPOCHS": 1,         # Epochs per training session (after each game)
    "BATCH_SIZE": 64,

    # -- MCTS Settings --
    # Phase AB: Increased from 400 to 800 to leverage parallel MCTS.
    "MCTS_SIMULATIONS": 800,      # Number of MCTS simulations per move
    "CPUCT": 1.25,                # Exploration constant in MCTS
    
    # -- Tactical Puzzle Settings --
    "TACTICAL_PUZZLE_FILENAME": "tactical_puzzles.jsonl",
    "PUZZLE_RATIO": 0.25, # Ratio of puzzles to mix into a standard training batch

    # -- Phase AR: Tactical Primer Settings --
    "TACTICAL_PRIMER_BATCHES": 3, # Number of batches for the tactical primer intervention.
    
    # -- Supervisor Parameters --
    'SUPERVISOR_WINDOW_SIZE': 20,
    'SUPERVISOR_PERFORMANCE_THRESHOLD': 7.0,
    
    # -- Bayesian Supervisor Specific --
    # FINALIZED: Set penalty to 0.8 based on comparative analysis
    'SUPERVISOR_BAYESIAN_PENALTY': 0.8,
    'SUPERVISOR_RECENCY_WINDOW': 50,
    # NOTE: Grace period logic is now handled in run_training.py by checking the last game type.
    # The parameter below is currently unused but kept for potential future logic.
    'SUPERVISOR_GRACE_PERIOD': 1, 

    # -- Mentor & Opponent Settings --
    # Phase AM: Increased Mentor Elo from 1350 to 2000.
    "MENTOR_ELO": 2000,
    "MENTOR_GAME_AGENT_COLOR": "random", # Color our agent plays in mentor games ("white", "black", or "random")
    "STOCKFISH_DEPTH_MENTOR": 10,       # Stockfish depth for mentor games
    "STOCKFISH_DEPTH_EVAL": 10,         # Stockfish depth for formal evaluation

    # -- Neural Network & Training Settings --
    "LEARNING_RATE": 0.0001,
    "WEIGHT_DECAY": 0.0001,

    # -- LR Scheduler Settings (Phase AG) --
    'LR_SCHEDULER_STEP_SIZE': 100,
    'LR_SCHEDULER_GAMMA': 0.9,
}


# =================================================================
# 2. Path Configuration (Colab-aware)
# =================================================================

Paths = namedtuple('Paths', [
    'checkpoints_dir', 
    'training_data_dir', 
    'pgn_games_dir', 
    'analysis_output_dir',
    'tactical_puzzles_file',
    'drive_project_root',
    'loss_log_file',
    'supervisor_log_file'
])

def get_paths():
    """
    Detects if running in Google Colab and returns a named tuple of appropriate
    paths for data, checkpoints, PGN files, and analysis outputs.
    """
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected. Using pre-mounted Google Drive paths.")
        
        drive_root_path = Path('/content/drive/MyDrive/ChessMCTS_RL')
        
        if not Path('/content/drive').is_dir():
                raise IOError(
                    "Google Drive is not mounted. Please mount it in a Colab cell "
                    "before running the script using: from google.colab import drive; "
                    "drive.mount('/content/drive')"
                )
            
    else:
        print("Running in a local environment.")
        # Assumes the script is run from the project root.
        drive_root_path = Path.cwd()
    
    checkpoints_path = drive_root_path / 'checkpoints'
    training_data_path = drive_root_path / 'training_data'
    pgn_games_path = drive_root_path / 'pgn_games'
    analysis_output_path = drive_root_path / 'analysis_output' # Typically for local analysis artifacts

    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)
    pgn_games_path.mkdir(parents=True, exist_ok=True)
    analysis_output_path.mkdir(parents=True, exist_ok=True)
    
    tactical_puzzles_path = drive_root_path / config_params["TACTICAL_PUZZLE_FILENAME"]
    loss_log_filepath = drive_root_path / 'loss_log_v2.csv'
    supervisor_log_filepath = drive_root_path / 'supervisor_log.txt'
    
    return Paths(
        checkpoints_dir=checkpoints_path,
        training_data_dir=training_data_path,
        pgn_games_dir=pgn_games_path,
        analysis_output_dir=analysis_output_path,
        tactical_puzzles_file=tactical_puzzles_path,
        drive_project_root=drive_root_path,
        loss_log_file=loss_log_filepath,
        supervisor_log_file=supervisor_log_filepath
    )