#
# File: tests/neural_network/test_policy_value_heads_phase_bi.py
#
import unittest
import torch
from gnn_agent.neural_network.policy_value_heads import ValueHead

class TestValueHeadPhaseBI(unittest.TestCase):
    def setUp(self):
        """Set up a device and some default parameters for tests."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = 256  # Example embedding dimension
        self.batch_size = 4      # Example batch size
        print(f"Running tests on {self.device}")

    def test_value_head_output_structure(self):
        """
        Tests that the modified ValueHead returns a tuple of two tensors.
        """
        value_head = ValueHead(embedding_dim=self.embedding_dim).to(self.device)
        
        # Create a dummy input tensor
        # Shape: (total_squares, embedding_dim)
        dummy_input = torch.randn(self.batch_size * 64, self.embedding_dim).to(self.device)
        
        # Create a dummy batch tensor
        dummy_batch = torch.repeat_interleave(torch.arange(self.batch_size), 64).to(self.device)

        # Perform forward pass
        outputs = value_head(dummy_input, dummy_batch)

        # 1. Check if the output is a tuple
        self.assertIsInstance(outputs, tuple, "ValueHead should return a tuple.")
        self.assertEqual(len(outputs), 2, "ValueHead should return exactly two outputs.")

    def test_value_head_output_shapes_and_types(self):
        """
        Tests the shape and dtype of the two output tensors from ValueHead.
        """
        value_head = ValueHead(embedding_dim=self.embedding_dim).to(self.device)
        dummy_input = torch.randn(self.batch_size * 64, self.embedding_dim).to(self.device)
        dummy_batch = torch.repeat_interleave(torch.arange(self.batch_size), 64).to(self.device)
        
        value, material_balance = value_head(dummy_input, dummy_batch)

        # 2. Check the 'value' output
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(value.shape, (self.batch_size, 1), "Game outcome value tensor has incorrect shape.")
        self.assertEqual(value.dtype, torch.float32)

        # 3. Check the 'material_balance' output
        self.assertIsInstance(material_balance, torch.Tensor)
        self.assertEqual(material_balance.shape, (self.batch_size, 1), "Material balance tensor has incorrect shape.")
        self.assertEqual(material_balance.dtype, torch.float32)

    def test_value_range(self):
        """
        Tests that the game outcome value is correctly squashed into the [-1, 1] range.
        """
        value_head = ValueHead(embedding_dim=self.embedding_dim).to(self.device)
        dummy_input = torch.randn(self.batch_size * 64, self.embedding_dim).to(self.device)
        dummy_batch = torch.repeat_interleave(torch.arange(self.batch_size), 64).to(self.device)
        
        value, _ = value_head(dummy_input, dummy_batch)

        # Check that all values are within the [-1, 1] range of tanh
        self.assertTrue(torch.all(value >= -1.0) and torch.all(value <= 1.0))

if __name__ == '__main__':
    unittest.main()
