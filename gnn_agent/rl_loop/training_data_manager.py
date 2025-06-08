import torch # We will use torch for saving/loading
from typing import List, Tuple, Dict
import chess
from pathlib import Path

# The TrainingData type definition remains the same
TrainingData = List[Tuple[torch.Tensor, Dict[chess.Move, float], float]]

class TrainingDataManager:
    """
    Manages saving and loading of self-play training data.
    """
    def __init__(self, data_directory: Path):
        """
        Initializes the data manager.
        """
        self.data_directory = data_directory
        self.data_directory.mkdir(parents=True, exist_ok=True)

    def save_data(self, data: TrainingData, filename: str):
        """
        Saves a list of training examples to a file using torch.save.
        This is the robust way to save data containing PyTorch tensors.

        Args:
            data (TrainingData): The training data from a self-play game.
            filename (str): The name of the file to save the data to.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        
        filepath = self.data_directory / filename
        
        # --- FIX: Use torch.save() instead of pickle.dump() ---
        torch.save(data, filepath)
        # --- END OF FIX ---
        
        print(f"Data saved to {filepath}")

    def load_data(self, filename: str) -> TrainingData:
        """
        Loads a list of training examples from a file using torch.load.

        Args:
            filename (str): The name of the file to load data from.
        
        Returns:
            TrainingData: The loaded training data.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
            
        filepath = self.data_directory / filename

        if not filepath.exists():
            print(f"Error: Data file not found at {filepath}")
            return []
            
        # --- FIX: Use torch.load() and include our robustness arguments ---
        # This makes the load function compatible with the save function
        # and also handles the CPU/GPU and PyTorch version issues we saw before.
        return torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        # --- END OF FIX ---