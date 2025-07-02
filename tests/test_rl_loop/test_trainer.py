# tests/rl_loop/test_trainer.py

import unittest
import torch
import chess
import tempfile
import shutil
import sys
from pathlib import Path

# --- FIX: Add project root to Python path ---
# This allows the script to be run directly and find the gnn_agent package.
# It navigates up from the test file's location (tests/rl_loop/test_trainer.py)
# to the project root (chess/).
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
# --- END FIX ---

from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters.gnn_data_converter import GNN_INPUT_FEATURE_DIM
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size


class TestTrainer(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for checkpoints and a trainer instance."""
        # Create a temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir()

        # Configuration for the trainer and network
        self.model_config = {
            'LEARNING_RATE': 0.01,
            'WEIGHT_DECAY': 0.0001,
            'LR_SCHEDULER_STEP_SIZE': 2,  # Step after 2 calls
            'LR_SCHEDULER_GAMMA': 0.5    # Halve the learning rate
        }

        # Create a trainer instance. The network will be initialized by load_or_initialize
        self.trainer = Trainer(model_config=self.model_config, device=torch.device("cpu"))

    def tearDown(self):
        """Remove the temporary directory after tests are done."""
        shutil.rmtree(self.temp_dir)

    def test_initialize_new_network(self):
        """
        Tests that a new network, optimizer, and scheduler are created correctly
        when no checkpoint exists.
        """
        network, game_number = self.trainer.load_or_initialize_network(self.checkpoint_dir)

        self.assertIsInstance(network, ChessNetwork)
        self.assertIsInstance(self.trainer.optimizer, torch.optim.Adam)
        self.assertIsInstance(self.trainer.scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(game_number, 0)
        self.assertEqual(self.trainer.optimizer.param_groups[0]['lr'], self.model_config['LEARNING_RATE'])

    def test_save_and_load_checkpoint(self):
        """
        Tests that a checkpoint can be saved and then loaded correctly,
        restoring the state of the model, optimizer, and scheduler.
        """
        # 1. Initialize and modify the state
        self.trainer.load_or_initialize_network(self.checkpoint_dir)
        initial_lr = self.trainer.optimizer.param_groups[0]['lr']

        # Step the scheduler to change its state.
        # We call a dummy optimizer.step() before scheduler.step() to avoid a UserWarning.
        self.trainer.optimizer.step()
        self.trainer.scheduler.step()
        self.trainer.optimizer.step()
        self.trainer.scheduler.step()
        lr_after_stepping = self.trainer.optimizer.param_groups[0]['lr']
        self.assertNotEqual(initial_lr, lr_after_stepping)

        # 2. Save the checkpoint
        checkpoint_path = self.checkpoint_dir / "test_checkpoint.pth.tar"
        self.trainer.save_checkpoint(self.checkpoint_dir, game_number=5, filename_override="test_checkpoint.pth.tar")
        self.assertTrue(checkpoint_path.exists())

        # 3. Create a new trainer and load from the checkpoint
        new_trainer = Trainer(model_config=self.model_config, device=torch.device("cpu"))
        network, game_number = new_trainer.load_or_initialize_network(self.checkpoint_dir)

        # 4. Assert that the state has been restored
        self.assertEqual(game_number, 5)
        self.assertEqual(new_trainer.scheduler.last_epoch, 2)
        loaded_lr = new_trainer.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(loaded_lr, lr_after_stepping)
        
        # Check that model weights are the same
        self.assertTrue(torch.equal(
            self.trainer.network.state_dict()['value_head.fc1.weight'],
            new_trainer.network.state_dict()['value_head.fc1.weight']
        ))

    def test_lr_scheduler_step_and_checkpointing(self):
        """
        Tests that the LR scheduler is stepped correctly by train_on_batch and its state is
        saved and loaded via checkpoints.
        """
        # 1. Initialize
        self.trainer.load_or_initialize_network(self.checkpoint_dir)

        # 2. Mock data for training calls
        board = chess.Board()
        move = list(board.legal_moves)[0]
        # The content doesn't matter, we just need to call the method with the right format
        mock_game_data = [(board.fen(), {move: 1.0}, 1.0)]
        mock_puzzle_data = [{'fen': board.fen(), 'best_move_uci': move.uci()}]

        # 3. Check initial LR
        initial_lr = self.trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(initial_lr, 0.01)

        # 4. Call train_on_batch once, LR should not change
        self.trainer.train_on_batch(mock_game_data, mock_puzzle_data, batch_size=32)
        lr_after_one_step = self.trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(lr_after_one_step, 0.01)
        self.assertEqual(self.trainer.scheduler.last_epoch, 1)

        # 5. Call train_on_batch again (total 2), LR should be halved
        self.trainer.train_on_batch(mock_game_data, mock_puzzle_data, batch_size=32)
        lr_after_two_steps = self.trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(lr_after_two_steps, 0.01 * 0.5)
        self.assertEqual(self.trainer.scheduler.last_epoch, 2)

        # 6. Save and load to ensure scheduler state persists
        self.trainer.save_checkpoint(self.checkpoint_dir, game_number=2)
        new_trainer = Trainer(model_config=self.model_config, device=torch.device("cpu"))
        new_trainer.load_or_initialize_network(self.checkpoint_dir)
        
        # 7. Verify the loaded state
        self.assertEqual(new_trainer.scheduler.last_epoch, 2)
        loaded_lr = new_trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(loaded_lr, 0.01 * 0.5)

        # 8. Perform one more step to ensure it continues correctly
        new_trainer.train_on_batch(mock_game_data, mock_puzzle_data, batch_size=32)
        lr_after_three_steps = new_trainer.optimizer.param_groups[0]['lr']
        self.assertEqual(lr_after_three_steps, 0.01 * 0.5) # Should not change yet
        self.assertEqual(new_trainer.scheduler.last_epoch, 3)

    def test_train_on_batch_runs_without_error(self):
        """
        A simple test to ensure train_on_batch runs with valid data
        without raising an exception. This confirms the complex data collation works.
        """
        self.trainer.load_or_initialize_network(self.checkpoint_dir)
        
        # Create some valid-looking data
        board1 = chess.Board()
        move1 = chess.Move.from_uci('e2e4')
        policy1 = {move1: 1.0}
        
        board2 = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        move2 = chess.Move.from_uci('g1f3')
        policy2 = {move2: 0.8, chess.Move.from_uci('b1c3'): 0.2}

        game_examples = [
            (board1.fen(), policy1, 1.0),
            (board2.fen(), policy2, -1.0)
        ]
        puzzle_examples = [
            {'fen': board1.fen(), 'best_move_uci': 'e2e4'}
        ]

        try:
            policy_loss, value_loss = self.trainer.train_on_batch(
                game_examples, puzzle_examples, batch_size=2, puzzle_ratio=0.5
            )
            # Check that we get valid loss values back
            self.assertIsInstance(policy_loss, float)
            self.assertIsInstance(value_loss, float)
            self.assertGreaterEqual(policy_loss, 0)
            self.assertGreaterEqual(value_loss, 0)
        except Exception as e:
            self.fail(f"train_on_batch raised an exception with valid data: {e}")

if __name__ == '__main__':
    unittest.main()
