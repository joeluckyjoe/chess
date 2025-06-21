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

    # --- ThresholdSupervisor Parameters ---
    'SUPERVISOR_WINDOW_SIZE': 20,       # Self-Play: Number of recent games to analyze for stagnation.
    'SUPERVISOR_VOLATILITY_THRESHOLD': 0.25, # Self-Play: Std deviation of policy loss to trigger mentor mode.
    'SUPERVISOR_PERFORMANCE_THRESHOLD': 1.8,  # Self-Play: Average policy loss to trigger mentor mode.
    'SUPERVISOR_GRADUATION_WINDOW': 10,       # Mentor-Play: Number of recent games to analyze for graduation.
    'SUPERVISOR_GRADUATION_THRESHOLD': 0.05,  # Mentor-Play: Avg value loss to trigger graduation back to self-play.
    'SUPERVISOR_MOVE_COUNT_THRESHOLD': 25,    # Mentor-Play: Avg number of moves to trigger graduation.
    
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
        pgn_games_path = base_drive_path / 'pgn_games' # <-- ADDED FOR COLAB
        
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
        pgn_games_path = base_path / 'pgn_games' # <-- ADDED FOR LOCAL

    # Create the directories if they don't exist
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)
    pgn_games_path.mkdir(parents=True, exist_ok=True) # <-- ADDED
    
    # MODIFIED: Return all three essential paths
    return checkpoints_path, training_data_path, pgn_games_path

