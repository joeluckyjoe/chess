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
    "TOTAL_GAMES": 2000,          # Total games to run in the training session
    "CHECKPOINT_INTERVAL": 10,    # Save a checkpoint every N games
    "TRAINING_EPOCHS": 1,         # Epochs per training session (after each game)
    "BATCH_SIZE": 64,

    # -- MCTS Settings --
    "MCTS_SIMULATIONS": 400,      # Number of MCTS simulations per move
    "CPUCT": 1.25,                # Exploration constant in MCTS

    # --- Supervisor Parameters ---
    'SUPERVISOR_WINDOW_SIZE': 20,
    'SUPERVISOR_PERFORMANCE_THRESHOLD': 7.0,
    
    # -- Bayesian Supervisor Specific --
    'SUPERVISOR_BAYESIAN_PENALTY': 2,
    'SUPERVISOR_RECENCY_WINDOW': 50, 
    'SUPERVISOR_GRACE_PERIOD': 10, # Games to wait after a mentor session

    # -- Mentor & Opponent Settings --
    "MENTOR_GAME_AGENT_COLOR": "random", # Color our agent plays in mentor games ("white", "black", or "random")
    "STOCKFISH_DEPTH_MENTOR": 10,        # Stockfish depth for mentor games
    "STOCKFISH_DEPTH_EVAL": 10,          # Stockfish depth for formal evaluation

    # -- Neural Network & Training Settings --
    "LEARNING_RATE": 0.0001,
    "WEIGHT_DECAY": 0.0001,
}


# =================================================================
# 2. Path Configuration (Colab-aware)
# =================================================================

# Define a named tuple for clean path access
Paths = namedtuple('Paths', [
    'checkpoints_dir', 
    'training_data_dir', 
    'pgn_games_dir', 
    'analysis_output_dir',
    'project_root'
])

def get_paths():
    """
    Detects if running in Google Colab and returns a named tuple of appropriate
    paths for data, checkpoints, PGN files, and analysis outputs.
    """
    # Check for a Colab environment variable
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected. Using pre-mounted Google Drive paths.")
        
        # In Colab, the project root is typically /content/chess, while data is on Drive
        project_root_path = Path('/content/chess')
        base_drive_path = Path('/content/drive/MyDrive/ChessMCTS_RL')
        
        checkpoints_path = base_drive_path / 'checkpoints'
        training_data_path = base_drive_path / 'training_data'
        pgn_games_path = base_drive_path / 'pgn_games'
        # Analysis output is local to the Colab instance for speed
        analysis_output_path = project_root_path / 'analysis_output'
        
        if not Path('/content/drive').is_dir():
                raise IOError(
                    "Google Drive is not mounted. Please mount it in a Colab cell "
                    "before running the script using: from google.colab import drive; "
                    "drive.mount('/content/drive')"
                )
            
    else:
        print("Running locally.")
        project_root_path = Path(__file__).resolve().parent
        
        checkpoints_path = project_root_path / 'checkpoints'
        training_data_path = project_root_path / 'training_data'
        pgn_games_path = project_root_path / 'pgn_games'
        analysis_output_path = project_root_path / 'analysis_output'

    # Create all necessary directories if they don't exist
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)
    pgn_games_path.mkdir(parents=True, exist_ok=True)
    analysis_output_path.mkdir(parents=True, exist_ok=True)
    
    # Return a named tuple for backwards-compatible access (by index)
    # and readable access (by name).
    return Paths(
        checkpoints_dir=checkpoints_path,
        training_data_dir=training_data_path,
        pgn_games_dir=pgn_games_path,
        analysis_output_dir=analysis_output_path,
        project_root=project_root_path
    )
