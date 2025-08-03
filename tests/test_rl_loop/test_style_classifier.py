import unittest
import chess
import torch
from pathlib import Path

# Add project root to path to allow importing from our modules
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from gnn_agent.rl_loop.style_classifier import StyleClassifier
from config import get_paths
from hardware_setup import get_device

class TestMLStyleClassifier(unittest.TestCase):

    def setUp(self):
        """
        Set up the classifier for an integration test.
        Note: This test requires the trained checkpoints to be available.
        """
        self.device = get_device()
        paths = get_paths()
        
        self.base_checkpoint_path = paths.checkpoints_dir / "br_checkpoint_game_1010.pth.tar"
        # --- MODIFIED: Point to the new middlegame classifier checkpoint ---
        self.classifier_head_path = paths.checkpoints_dir / "best_middlegame_classifier.pth"
        
        if not self.base_checkpoint_path.exists() or not self.classifier_head_path.exists():
            self.skipTest("Required model checkpoints not found. Skipping integration test.")
            
        self.classifier = StyleClassifier(
            base_checkpoint_path=self.base_checkpoint_path,
            classifier_head_path=self.classifier_head_path,
            device=self.device
        )

    def test_initialization(self):
        """Test if the classifier and its models initialize without errors."""
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.classifier.base_model)
        self.assertIsNotNone(self.classifier.classifier_head)

    def test_score_move_returns_valid_reward(self):
        """
        Test that score_move returns a float within the expected range [-0.5, 0.5].
        """
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        
        reward = self.classifier.score_move(board, move)
        
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, -0.5)
        self.assertLessEqual(reward, 0.5)
        print(f"Sample reward for e4: {reward:.4f}")

if __name__ == '__main__':
    unittest.main()