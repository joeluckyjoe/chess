import unittest
import torch
from torch_geometric.data import HeteroData

from gnn_agent.neural_network.hybrid_rnn_model import HybridRNNModel
# The GNNDataConverter import is removed as it does not exist.

class TestHybridRNNModel(unittest.TestCase):

    def setUp(self):
        """Set up a dummy model and sample data for testing."""
        self.gnn_hidden_dim = 64
        self.cnn_in_channels = 14
        self.embed_dim = 256
        self.num_heads = 4
        self.rnn_hidden_dim = 512
        self.num_rnn_layers = 2
        self.policy_size = 4672
        
        # --- FIX ---
        # The GNNDataConverter class doesn't exist.
        # We will manually define the metadata tuple that the model expects.
        # This tuple consists of (node_types, edge_types).
        self.gnn_metadata = (
            ['square', 'piece'],  # Node types
            [  # Edge types
                ('piece', 'occupies', 'square'),
                ('piece', 'attacks', 'piece'),
                ('piece', 'defends', 'piece'),
                ('square', 'adjacent_to', 'square')
            ]
        )

        self.model = HybridRNNModel(
            gnn_hidden_dim=self.gnn_hidden_dim,
            cnn_in_channels=self.cnn_in_channels,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            gnn_metadata=self.gnn_metadata,
            rnn_hidden_dim=self.rnn_hidden_dim,
            num_rnn_layers=self.num_rnn_layers,
            policy_size=self.policy_size
        )

    def _create_dummy_hetero_data(self):
        """Creates a single dummy HeteroData object."""
        data = HeteroData()
        data['square'].x = torch.randn(64, 13) # Dim from gnn_data_converter.py
        data['piece'].x = torch.randn(32, 22)  # Dim from gnn_data_converter.py
        
        # Define edge types consistent with metadata
        data['piece', 'occupies', 'square'].edge_index = torch.stack([torch.randint(0, 32, (32,)), torch.arange(32)], dim=0)
        data['piece', 'attacks', 'piece'].edge_index = torch.randint(0, 32, (2, 50))
        data['piece', 'defends', 'piece'].edge_index = torch.randint(0, 32, (2, 40))
        data['square', 'adjacent_to', 'square'].edge_index = torch.randint(0, 64, (2, 200))
        return data

    def test_forward_pass_sequence(self):
        """Test a forward pass with a sequence of inputs, validating the architecture."""
        seq_len = 5
        batch_size = 1

        gnn_data_seq = [self._create_dummy_hetero_data() for _ in range(seq_len)]
        cnn_input_seq = torch.randn(batch_size, seq_len, self.cnn_in_channels, 8, 8)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, value, final_hidden = self.model(gnn_data_seq, cnn_input_seq)
        
        self.assertEqual(policy_logits.shape, (batch_size, self.policy_size))
        self.assertEqual(value.shape, (batch_size, 1))
        self.assertEqual(final_hidden.shape, (self.num_rnn_layers, batch_size, self.rnn_hidden_dim))

if __name__ == '__main__':
    unittest.main()