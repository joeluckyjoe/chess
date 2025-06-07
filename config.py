import os
from pathlib import Path

# =================================================================
# 1. Hyperparameter Configuration
# =================================================================

config_params = {
    # -- General & Path Settings --
    "DEVICE": "auto",  # Use "auto" to detect CUDA, or force "cpu"
    "STOCKFISH_PATH": "/usr/games/stockfish",

    # -- Self-Play & MCTS Settings --
    # --- MODIFIED: Increased game count for the second training run ---
    "NUM_SELF_PLAY_GAMES": 2000,
    "CHECKPOINT_INTERVAL": 10,  # Save a checkpoint every N games
    "MCTS_SIMULATIONS": 50,     # Number of MCTS simulations per move
    "CPUCT": 1.25,              # Exploration constant in MCTS

    # -- Neural Network & Training Settings --
    "LEARNING_RATE": 0.0001,  # LEARNING_RATE changed from 0.001 to 0.0001
    "WEIGHT_DECAY": 0.0001,   # L2 regularization
    "TRAINING_EPOCHS": 1,     # Epochs per training session (after each game)
    "BATCH_SIZE": 64,

    # -- Evaluation Settings --
    "EVALUATION_GAMES": 20,
    "EVALUATION_STOCKFISH_DEPTH": 5,
}


# =================================================================
# 2. Path Configuration (Colab-aware)
# =================================================================

def get_paths():
    """
    Detects if running in Google Colab and returns appropriate paths for data and checkpoints.
    
    This version ASSUMES that if running in Colab, Google Drive has already been
    mounted to /content/drive in an interactive notebook cell.
    
    Returns:
        tuple: A tuple containing (checkpoints_path, training_data_path)
    """
    # Check for a Colab environment variable
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected. Using pre-mounted Google Drive paths.")
        
        # We assume the drive is already mounted at /content/drive
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
        # Use local paths relative to the project root
        base_path = Path(__file__).resolve().parent
        checkpoints_path = base_path / 'checkpoints'
        training_data_path = base_path / 'training_data'

    # Create the directories if they don't exist
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)
    
    return checkpoints_path, training_data_path