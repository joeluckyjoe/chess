import unittest
import pickle
import torch
import chess
from pathlib import Path
import tempfile
import shutil

# Adjust the import path based on your project structure
from gnn_agent.rl_loop.training_data_manager import TrainingDataManager

class TestTrainingDataManager(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for test data."""
        self.test_dir = tempfile.mkdtemp()
        self.data_manager = TrainingDataManager(Path(self.test_dir))

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def test_save_and_load_data(self):
        """
        Test that data saved to a file can be loaded back correctly and is identical.
        """
        # 1. Create dummy training data
        move1 = chess.Move.from_uci("e2e4")
        move2 = chess.Move.from_uci("d2d4")
        original_data = [
            (torch.tensor([1.0, 0.0]), {move1: 0.9, move2: 0.1}, 1.0),
            (torch.tensor([0.0, 1.0]), {move2: 0.8, move1: 0.2}, -1.0),
        ]
        filename = "test_game_01.pkl"

        # 2. Save the data
        self.data_manager.save_data(original_data, filename)

        # Check that the file was actually created
        self.assertTrue((Path(self.test_dir) / filename).exists())

        # 3. Load the data
        loaded_data = self.data_manager.load_data(filename)

        # 4. Assert that the loaded data is identical to the original
        self.assertEqual(len(loaded_data), len(original_data))
        
        # Check first element
        self.assertTrue(torch.equal(loaded_data[0][0], original_data[0][0]))
        self.assertEqual(loaded_data[0][1], original_data[0][1])
        self.assertEqual(loaded_data[0][2], original_data[0][2])

        # Check second element
        self.assertTrue(torch.equal(loaded_data[1][0], original_data[1][0]))
        self.assertEqual(loaded_data[1][1], original_data[1][1])
        self.assertEqual(loaded_data[1][2], original_data[1][2])

    def test_filename_extension(self):
        """Test that the .pkl extension is added if missing."""
        original_data = [(torch.tensor([1.0]), {}, 1.0)]
        base_filename = "test_game_02"
        
        # Save without extension
        self.data_manager.save_data(original_data, base_filename)
        
        # Check that file with .pkl extension exists
        self.assertTrue((Path(self.test_dir) / f"{base_filename}.pkl").exists())
        
        # Load without extension
        loaded_data = self.data_manager.load_data(base_filename)
        self.assertEqual(len(loaded_data), 1)

if __name__ == '__main__':
    unittest.main()