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
    # Phase AB: Increased from 400 to 800 to leverage parallel MCTS.
    "MCTS_SIMULATIONS": 800,      # Number of MCTS simulations per move
    "CPUCT": 1.25,                # Exploration constant in MCTS
    
    # --- Tactics Training Settings (Phase O) ---
    "TACTICS_SESSION_FREQUENCY": 20, # Run a tactics session every N games.
    "TACTICAL_PUZZLE_FILENAME": "tactical_puzzles.jsonl",

    # --- Supervisor Parameters ---
    'SUPERVISOR_WINDOW_SIZE': 20,
    'SUPERVISOR_PERFORMANCE_THRESHOLD': 7.0,
    
    # -- Bayesian Supervisor Specific --
    # FINALIZED: Set penalty to 0.8 based on comparative analysis
    'SUPERVISOR_BAYESIAN_PENALTY': 0.8,
    'SUPERVISOR_RECENCY_WINDOW': 50, 
    'SUPERVISOR_GRACE_PERIOD': 10, # Games to wait after a mentor session
    'MAX_INTERVENTION_GAMES': 5,   # NEW: Max games in a mentor burst

    # -- Mentor & Opponent Settings --
    # Phase AM: Increased Mentor Elo from 1350 to 2000.
    "MENTOR_ELO": 2000,
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

Paths = namedtuple('Paths', [
    'checkpoints_dir', 
    'training_data_dir', 
    'pgn_games_dir', 
    'analysis_output_dir',
    'tactical_puzzles_file',
    'local_project_root',
    'drive_project_root'
])

def get_paths():
    """
    Detects if running in Google Colab and returns a named tuple of appropriate
    paths for data, checkpoints, PGN files, and analysis outputs.
    """
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected. Using pre-mounted Google Drive paths.")
        
        local_root_path = Path('/content/chess')
        drive_root_path = Path('/content/drive/MyDrive/ChessMCTS_RL')
        
        checkpoints_path = drive_root_path / 'checkpoints'
        training_data_path = drive_root_path / 'training_data'
        pgn_games_path = drive_root_path / 'pgn_games'
        analysis_output_path = local_root_path / 'analysis_output'
        
        if not Path('/content/drive').is_dir():
                raise IOError(
                    "Google Drive is not mounted. Please mount it in a Colab cell "
                    "before running the script using: from google.colab import drive; "
                    "drive.mount('/content/drive')"
                )
            
    else:
        print("Running locally.")
        local_root_path = Path(__file__).resolve().parent
        drive_root_path = local_root_path
        
        checkpoints_path = local_root_path / 'checkpoints'
        training_data_path = local_root_path / 'training_data'
        pgn_games_path = local_root_path / 'pgn_games'
        analysis_output_path = local_root_path / 'analysis_output'

    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)
    pgn_games_path.mkdir(parents=True, exist_ok=True)
    analysis_output_path.mkdir(parents=True, exist_ok=True)
    
    tactical_puzzles_path = drive_root_path / config_params["TACTICAL_PUZZLE_FILENAME"]
    
    return Paths(
        checkpoints_dir=checkpoints_path,
        training_data_dir=training_data_path,
        pgn_games_dir=pgn_games_path,
        analysis_output_dir=analysis_output_path,
        tactical_puzzles_file=tactical_puzzles_path,
        local_project_root=local_root_path,
        drive_project_root=drive_root_path
    )