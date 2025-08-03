import os
from pathlib import Path
from collections import namedtuple

# =================================================================
# 1. Hyperparameter Configuration (Phase C: Expert Sparring)
# =================================================================

config_params = {
    # -- General & Path Settings --
    "DEVICE": "auto", # Options: "auto", "cpu", "cuda"
    "STOCKFISH_PATH": "/usr/games/stockfish", # Or your local path to the Stockfish executable

    # -- Expert Sparring Run Settings --
    "TOTAL_GAMES": 2500, # Total number of games to play for the entire run
    "CHECKPOINT_INTERVAL": 20, # Save a model checkpoint every N games
    "BATCH_SIZE": 256, # Number of game states to use for a single training update

    # -- Stockfish Opponent Settings --
    # MODIFIED: Increasing ELO to the strong club player level.
    "STOCKFISH_ELO": 1400,
    "STOCKFISH_DEPTH": 5,

    # -- MCTS Settings --
    # MODIFIED: Increasing simulations to find better moves against a stronger opponent.
    "MCTS_SIMULATIONS": 400,
    "CPUCT": 4.0, # Exploration-exploitation trade-off in PUCT formula

    # -- Neural Network Architecture --
    "EMBED_DIM": 256, # Dimension of the feature embeddings
    "GNN_HIDDEN_DIM": 128, # Hidden dimension within the GNN layers
    "GNN_NUM_HEADS": 4, # Number of attention heads in the GNN

    # -- Optimizer & Learning Rate --
    "LEARNING_RATE": 0.00002,
    "WEIGHT_DECAY": 0.0001,
}


# =================================================================
# 2. Path Configuration (Colab-aware)
# =================================================================
# This section is unchanged.

Paths = namedtuple('Paths', [
    'checkpoints_dir',
    'pgn_games_dir',
    'drive_project_root',
])

def get_paths():
    """
    Determines if running in Colab or local environment and returns
    a named tuple with the correct, absolute paths.
    """
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
    pgn_games_path = drive_root_path / 'pgn_games'

    checkpoints_path.mkdir(parents=True, exist_ok=True)
    pgn_games_path.mkdir(parents=True, exist_ok=True)

    return Paths(
        checkpoints_dir=checkpoints_path,
        pgn_games_dir=pgn_games_path,
        drive_project_root=drive_root_path,
    )