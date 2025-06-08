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
    "TOTAL_GAMES": 2000,          # Total games to run in the training session (increased for full run)
    "CHECKPOINT_INTERVAL": 50,    # Save a checkpoint every N games (adjusted for longer run)
    "TRAINING_EPOCHS": 1,         # Epochs per training session (after each game)
    "BATCH_SIZE": 64,

    # -- MCTS Settings --
    "MCTS_SIMULATIONS": 400,      # Number of MCTS simulations per move (increased for better move quality)
    "CPUCT": 1.25,                # Exploration constant in MCTS

    # -- Mentor-Guided Training Settings (NEW) --
    "MENTOR_GAME_INTERVAL": 5,        # Play a mentor game every 5 games (1 mentor, 4 self-play)
    "MENTOR_GAME_AGENT_COLOR": "random", # Color our agent plays in mentor games ("white", "black", or "random")

    # -- Opponent Settings --
    "STOCKFISH_DEPTH_MENTOR": 10,     # Stockfish depth for mentor games (increased for stronger guidance)
    "STOCKFISH_DEPTH_EVAL": 10,       # Stockfish depth for formal evaluation (matched to mentor)

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
