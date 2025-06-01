# tests/test_rl_loop/test_trainer.py

import unittest
from unittest.mock import MagicMock, patch
import torch

from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.neural_network.chess_network import ChessNetwork

class TestTrainer(unittest.TestCase):

    def setUp(self):
        """Set up a mock network and trainer for tests."""
        # Mock the ChessNetwork
        self.mock_network = MagicMock(spec=ChessNetwork)
        # Make the mock network's parameters trackable by the optimizer
        self.mock_network.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]

        # Configure the forward pass mock to return tensors of the correct shape
        # Batch size = 2, policy output size = 4672 (standard for chess), value output size = 1
        self.mock_network.return_value = (
            torch.randn(2, 4672), # Mock policy logits
            torch.randn(2, 1)     # Mock values
        )

        self.trainer = Trainer(self.mock_network, learning_rate=0.01)

    def test_train_on_batch_executes_training_step(self):
        """
        Verify that train_on_batch performs a forward pass, backward pass,
        and optimizer step.
        """
        # Create a batch of mock data (batch size = 2)
        mock_batch_data = [
            (torch.randn(1, 8, 8), torch.randn(4672), 1.0),
            (torch.randn(1, 8, 8), torch.randn(4672), -1.0)
        ]

        # Patch the optimizer to track its calls
        with patch.object(self.trainer.optimizer, 'zero_grad') as mock_zero_grad, \
             patch.object(self.trainer.optimizer, 'step') as mock_step:

            # Mock the backward call on the resulting loss tensor
            # We can't easily mock the loss tensor itself, so we mock what's called on it.
            # Since total_loss is a live tensor, we patch the 'backward' method on the tensor type.
            with patch.object(torch.Tensor, 'backward') as mock_backward:
                policy_loss, value_loss = self.trainer.train_on_batch(mock_batch_data)

                # 1. Verify network was called (forward pass)
                self.mock_network.assert_called_once()
                self.assertEqual(self.mock_network.call_args[0][0].shape[0], 2) # Batch size of 2

                # 2. Verify loss calculation and backward pass
                self.assertIsNotNone(policy_loss)
                self.assertIsNotNone(value_loss)
                mock_backward.assert_called_once()

                # 3. Verify optimizer steps
                mock_zero_grad.assert_called_once()
                mock_step.assert_called_once()

    def test_train_on_empty_batch(self):
        """
        Test that training on an empty batch does not raise an error and returns None.
        """
        policy_loss, value_loss = self.trainer.train_on_batch([])
        self.assertIsNone(policy_loss)
        self.assertIsNone(value_loss)
        self.mock_network.assert_not_called()

if __name__ == '__main__':
    unittest.main()