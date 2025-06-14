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

    # -- Training Supervisor Settings (Tuned for Local Test) --
    "supervisor_loss_history_size": 20, # Number of self-play games to analyze for stagnation (LOWERED for local test)
    "stagnation_window": 0.25,          # % of the loss history to check for a recent changepoint
    "ruptures_model": "l2",             # Model used by the ruptures library for changepoint detection
    "ruptures_penalty": 1,              # (TUNED VALUE) Penalty for creating changepoints.
    
    "mentor_history_size": 5,           # Number of mentor games to analyze for improvement
    "mentor_win_threshold": 1,          # Switch to self-play after this many WINS against mentor
    "mentor_draw_threshold": 2,         # Switch to self-play after this many DRAWS against mentor

    # -- Mentor & Opponent Settings --
    "MENTOR_GAME_AGENT_COLOR": "random", # Color our agent plays in mentor games ("white", "black", or "random")
    "STOCKFISH_DEPTH_MENTOR": 10,       # Stockfish depth for mentor games
    "STOCKFISH_DEPTH_EVAL": 10,         # Stockfish depth for formal evaluation

    # -- Neural Network & Training Settings --
    "LEARNING_RATE": 0.0001,
    "WEIGHT_DECAY": 0.0001,
}


# =================================================================
# 2. Path Configuration (Colab-aware)
# =================================================================

def get_paths():
    """
    Detects if running in Google Colab and returns appropriate paths for data and checkpoints.
    """
    # Check for a Colab environment variable
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected. Using pre-mounted Google Drive paths.")
        
        base_drive_path = Path('/content/drive/MyDrive/ChessMCTS_RL')
        checkpoints_path = base_drive_path / 'checkpoints'
        training_data_path = base_drive_path / 'training_data'
        
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

    # Create the directories if they don't exist
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)
    
    return checkpoints_path, training_data_path
