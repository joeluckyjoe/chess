import unittest
import torch
from torch_geometric.data import HeteroData, Batch

# Adjust the import path based on the project structure
from gnn_agent.neural_network.policy_value_model import PolicyValueModel
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size
from gnn_agent.gamestate_converters.gnn_data_converter import SQUARE_FEATURE_DIM, PIECE_FEATURE_DIM

class TestPolicyValueModel(unittest.TestCase):

    def setUp(self):
        """Set up a dummy configuration and model for testing."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # This metadata must match what the UnifiedGNN expects
        self.gnn_metadata = (
            ['square', 'piece'],
            [
                ('square', 'adjacent_to', 'square'),
                ('piece', 'occupies', 'square'),
                ('piece', 'attacks', 'piece'),
                ('piece', 'defends', 'piece')
            ]
        )
        
        self.model_config = {
            'gnn_hidden_dim': 128,
            'cnn_in_channels': 14,
            'embed_dim': 256,
            'policy_size': get_action_space_size(),
            'gnn_num_heads': 4,
            'gnn_metadata': self.gnn_metadata
        }
        
        self.model = PolicyValueModel(**self.model_config).to(self.device)
        self.batch_size = 4

    def _create_dummy_gnn_batch(self):
        """Creates a dummy batch of heterogeneous graph data with correct feature dimensions and all edge types."""
        data_list = []
        for _ in range(self.batch_size):
            data = HeteroData()
            
            data['square'].x = torch.randn(64, SQUARE_FEATURE_DIM) 
            data['piece'].x = torch.randn(32, PIECE_FEATURE_DIM)
            
            # --- Provide dummy data for ALL edge types defined in metadata ---
            data['square', 'adjacent_to', 'square'].edge_index = torch.randint(0, 64, (2, 100), dtype=torch.long)
            
            piece_occupies_square_edges = torch.cat([
                torch.randint(0, 32, (1, 32)),
                torch.randint(0, 64, (1, 32))
            ], dim=0)
            data['piece', 'occupies', 'square'].edge_index = piece_occupies_square_edges.to(torch.long)
            
            # Add dummy data for the missing edge types
            data['piece', 'attacks', 'piece'].edge_index = torch.randint(0, 32, (2, 50), dtype=torch.long)
            data['piece', 'defends', 'piece'].edge_index = torch.randint(0, 32, (2, 50), dtype=torch.long)

            data_list.append(data)
            
        return Batch.from_data_list(data_list).to(self.device)

    def test_forward_pass_shapes(self):
        """Test the output shapes of the forward pass."""
        dummy_gnn_batch = self._create_dummy_gnn_batch()
        dummy_cnn_tensor = torch.randn(self.batch_size, self.model_config['cnn_in_channels'], 8, 8).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(dummy_gnn_batch, dummy_cnn_tensor)
        
        self.assertEqual(policy_logits.shape, (self.batch_size, self.model_config['policy_size']))
        self.assertEqual(value.shape, (self.batch_size, 1))

    def test_model_initialization(self):
        """Test if the model and its components are initialized correctly."""
        self.assertIsInstance(self.model, PolicyValueModel)
        self.assertTrue(hasattr(self.model, 'gnn'))
        self.assertTrue(hasattr(self.model, 'cnn'))
        self.assertTrue(hasattr(self.model, 'policy_head'))
        self.assertTrue(hasattr(self.model, 'value_head'))
        self.assertFalse(hasattr(self.model, 'next_state_value_head'))

if __name__ == '__main__':
    unittest.main()