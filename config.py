# config.py

import sys
import os
from pathlib import Path

def get_paths():
    """
    Detects if running in Google Colab and returns appropriate paths for data and checkpoints.

    If in Colab, it mounts Google Drive and sets paths to a 'ChessMCTS_RL' folder
    within Drive. If not in Colab, it uses local paths.

    Returns:
        tuple: A tuple containing (checkpoints_path, training_data_path)
    """
    IN_COLAB = 'google.colab' in sys.modules

    if IN_COLAB:
        print("Running in Google Colab. Mounting Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')

        # Define a base path in your Google Drive
        base_drive_path = Path('/content/drive/MyDrive/ChessMCTS_RL')
        checkpoints_path = base_drive_path / 'checkpoints'
        training_data_path = base_drive_path / 'training_data'

        print(f"Data will be saved to: {base_drive_path}")

    else:
        print("Running locally.")
        # Use local paths relative to the project root
        base_path = Path(__file__).parent
        checkpoints_path = base_path / 'checkpoints'
        training_data_path = base_path / 'training_data'

    # Create the directories if they don't exist
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    training_data_path.mkdir(parents=True, exist_ok=True)

    return checkpoints_path, training_data_path