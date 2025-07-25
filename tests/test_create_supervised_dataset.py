# FILENAME: test_create_supervised_dataset.py

import unittest
import os
import json
from pathlib import Path
import shutil

# This import assumes the script is in the same directory or project root.
# Adjust the import path if your file structure is different.
from create_supervised_dataset import create_kasparov_dataset

class TestCreateSupervisedDataset(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and a dummy PGN file for testing."""
        self.test_dir = Path("test_temp_data")
        self.test_dir.mkdir(exist_ok=True)
        
        self.kasparov_pgn_path = self.test_dir / "Kasparov.pgn"
        self.output_dataset_path = self.test_dir / "kasparov_supervised_dataset.jsonl"

        # A simple PGN with one game won by White and another drawn
        pgn_content = """
[Event "Test Game 1"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. f4 exf4 3. Nf3 1-0

[Event "Test Game 2"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "Player C"]
[Black "Player D"]
[Result "1/2-1/2"]

1. d4 d5 1/2-1/2
"""
        with open(self.kasparov_pgn_path, "w") as f:
            f.write(pgn_content)
            
        # Mock the get_paths function to use our temporary directory
        self.original_get_paths = __import__('create_supervised_dataset').get_paths
        __import__('create_supervised_dataset').get_paths = self.mock_get_paths

    def mock_get_paths(self):
        """A mock version of get_paths that points to the temporary test directory."""
        from collections import namedtuple
        Paths = namedtuple('Paths', ['drive_project_root'])
        return Paths(drive_project_root=self.test_dir)

    def tearDown(self):
        """Clean up the temporary directory and its contents."""
        __import__('create_supervised_dataset').get_paths = self.original_get_paths
        shutil.rmtree(self.test_dir)

    def test_dataset_creation_and_content(self):
        """Test the creation of the dataset and verify its contents."""
        create_kasparov_dataset()

        # 1. Check if the output file was created
        self.assertTrue(self.output_dataset_path.exists())

        # 2. Read and verify the contents
        with open(self.output_dataset_path, 'r') as f:
            lines = f.readlines()
        
        # Game 1 has 5 moves (ply). Game 2 has 2 moves (ply). Total = 7 positions.
        self.assertEqual(len(lines), 7)

        # 3. Spot check a few data points
        
        # Game 1, Position 1 (White to move, White wins)
        pos1 = json.loads(lines[0])
        self.assertEqual(pos1['fen'], 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.assertEqual(pos1['played_move'], 'e2e4')
        self.assertEqual(pos1['outcome'], 1.0) # White's perspective, White won

        # Game 1, Position 2 (Black to move, White wins)
        pos2 = json.loads(lines[1])
        self.assertEqual(pos2['fen'], 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')
        self.assertEqual(pos2['played_move'], 'e7e5')
        self.assertEqual(pos2['outcome'], -1.0) # Black's perspective, White won

        # Game 2, Position 1 (White to move, Draw)
        pos6 = json.loads(lines[5])
        self.assertEqual(pos6['fen'], 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.assertEqual(pos6['played_move'], 'd2d4')
        self.assertEqual(pos6['outcome'], 0.0) # White's perspective, Draw

        # Game 2, Position 2 (Black to move, Draw)
        pos7 = json.loads(lines[6])
        self.assertEqual(pos7['fen'], 'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1')
        self.assertEqual(pos7['played_move'], 'd7d5')
        self.assertEqual(pos7['outcome'], 0.0) # Black's perspective, Draw


if __name__ == '__main__':
    unittest.main()