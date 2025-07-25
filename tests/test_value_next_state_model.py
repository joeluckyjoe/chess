# tests/test_value_next_state_model.py
import unittest
import torch
from torch_geometric.data import HeteroData, Batch

# Adjust the import path according to your project structure
from gnn_agent.neural_network.value_next_state_model import ValueNextStateModel
from gnn_agent.gamestate_converters.gnn_data_converter import PIECE_FEATURE_DIM, SQUARE_FEATURE_DIM

class TestValueNextStateModel(unittest.TestCase):

    def setUp(self):
        """Set up a dummy model and dummy data for testing."""
        self.model_config = {
            "gnn_hidden_dim": 128,
            "cnn_in_channels": 14,
            "embed_dim": 256,
            "policy_size": 4672,
            "gnn_num_heads": 4,
            "gnn_metadata": (
                ['piece', 'square'],
                [
                    ('piece', 'occupies', 'square'),
                    ('square', 'rev_occupies', 'piece'),
                    ('piece', 'attacks', 'piece'),
                    ('piece', 'defends', 'piece')
                ]
            )
        }
        self.model = ValueNextStateModel(**self.model_config)

        # --- Create Dummy Data ---
        self.batch_size = 4
        
        # GNN Data
        data_list = []
        for _ in range(self.batch_size):
            data = HeteroData()
            data['piece'].x = torch.randn(32, PIECE_FEATURE_DIM)
            data['square'].x = torch.randn(64, SQUARE_FEATURE_DIM)
            # Add dummy edge indices for all edge types
            for edge_type in self.model_config["gnn_metadata"][1]:
                num_src_nodes = 32 if edge_type[0] == 'piece' else 64
                num_dst_nodes = 32 if edge_type[2] == 'piece' else 64
                num_edges = 50 # Arbitrary number of edges
                data[edge_type].edge_index = torch.randint(0, min(num_src_nodes, num_dst_nodes), (2, num_edges), dtype=torch.long)
            data_list.append(data)
        
        self.gnn_batch = Batch.from_data_list(data_list)
        
        # CNN Data
        self.cnn_tensor = torch.randn(self.batch_size, self.model_config["cnn_in_channels"], 8, 8)

    def test_model_initialization(self):
        """Test if the model and its components are initialized."""
        self.assertIsInstance(self.model, ValueNextStateModel)
        self.assertTrue(hasattr(self.model, 'gnn'))
        self.assertTrue(hasattr(self.model, 'cnn'))
        self.assertTrue(hasattr(self.model, 'policy_head'))
        self.assertTrue(hasattr(self.model, 'value_head'))
        self.assertTrue(hasattr(self.model, 'next_state_value_head'))
        # Ensure the transformer is gone
        self.assertFalse(hasattr(self.model, 'transformer_encoder'))

    def test_forward_pass_shapes(self):
        """Test the output shapes of the forward pass."""
        self.model.eval()
        with torch.no_grad():
            policy_logits, value, next_state_value = self.model(self.gnn_batch, self.cnn_tensor)

        self.assertEqual(policy_logits.shape, (self.batch_size, self.model_config["policy_size"]))
        self.assertEqual(value.shape, (self.batch_size, 1))
        self.assertEqual(next_state_value.shape, (self.batch_size, 1))

    def test_value_output_range(self):
        """Test that value outputs are within the [-1, 1] range due to tanh."""
        self.model.eval()
        with torch.no_grad():
            _, value, next_state_value = self.model(self.gnn_batch, self.cnn_tensor)
        
        self.assertTrue(torch.all(value >= -1) and torch.all(value <= 1))
        self.assertTrue(torch.all(next_state_value >= -1) and torch.all(next_state_value <= 1))

if __name__ == '__main__':
    unittest.main()