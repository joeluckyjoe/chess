import pickle
from typing import List, Tuple, Dict
import chess
import torch
from pathlib import Path

TrainingData = List[Tuple[torch.Tensor, Dict[chess.Move, float], float]]

class TrainingDataManager:
    """
    Manages saving and loading of self-play training data.
    """
    def __init__(self, data_directory: Path):
        """
        Initializes the data manager.

        Args:
            data_directory (Path): The directory to store training data files.
        """
        self.data_directory = data_directory
        self.data_directory.mkdir(parents=True, exist_ok=True)

    def clear_data(self):
        """Deletes the training data file from the data directory."""
        # NOTE: This assumes the data file is named 'self_play_data.pkl' inside the directory.
        # Please verify this matches the filename used in your save_data() method.
        file_to_delete = self.data_directory / "self_play_data.pkl"

        try:
            if file_to_delete.exists():
                file_to_delete.unlink()
                # Optional: Add a log to confirm clearance
                # logging.info(f"Cleared old training data at {file_to_delete}")
        except OSError as e:
            # Optional: Add error logging
            # logging.error(f"Error clearing data file {file_to_delete}: {e}")
            raise e
         
    def save_data(self, data: TrainingData, filename: str):
        """
        Saves a list of training examples to a file using pickle.

        Args:
            data (TrainingData): The training data generated from a self-play game.
            filename (str): The name of the file to save the data to.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        
        filepath = self.data_directory / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filepath}")

    def load_data(self, filename: str) -> TrainingData:
        """
        Loads a list of training examples from a pickle file.

        Args:
            filename (str): The name of the file to load data from.
        
        Returns:
            TrainingData: The loaded training data.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
            
        filepath = self.data_directory / filename
        with open(filepath, 'rb') as f:
            return pickle.load(f)