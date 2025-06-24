import os
from pathlib import Path

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
    # These parameters are used by both the Statistical and Bayesian supervisors.
    'SUPERVISOR_WINDOW_SIZE': 20,
    'SUPERVISOR_PERFORMANCE_THRESHOLD': 7.0,
    
    # -- Statistical Supervisor Specific --
    'SUPERVISOR_P_VALUE_THRESHOLD': 0.05,
    
    # -- Bayesian Supervisor Specific --
    # This penalty value was determined after comparative analysis.
    'SUPERVISOR_BAYESIAN_PENALTY': 2, # <-- ADDED

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

def get_paths():
    """
    Detects if running in Google Colab and returns appropriate paths for data,
    checkpoints, and PGN files.
    """
    # Check for a Colab environment variable
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected. Using pre-mounted Google Drive paths.")
        
        base_drive_path = Path('/content/drive/MyDrive/ChessMCTS_RL')
        checkpoints_path = base_drive_path / 'checkpoints'
        training_data_path = base_drive_path / 'training_data'
        pgn_games_path = base_drive_path / 'pgn_games'
        
        if not Path('/content/drive').is_dir():
                raise IOError(
                    "Google Drive is not mounted. Please mount it in a Colab cell "
                    "before running the script using: from google.colab import drive; "
                    "drive.mount('/content/drive')"
                )
            
    else:
        print("Running locally.")
        base_path = Path(__file__).resolve().parent
        checkpoints_path = base_path / 'checkpoints'
        training_data_path = base_path / 'training_data'
        pgn_games_path = base_path / 'pgn_games'

    # Create the directories if they don't exist
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)
    pgn_games_path.mkdir(parents=True, exist_ok=True)
    
    return checkpoints_path, training_data_path, pgn_games_path
