import unittest
import torch
import numpy as np
import torch.nn.functional as F

# Assuming gnn_sq_input.py and square_features.py are accessible for input dimensions
from gnn_sq_input import NUM_SQUARES, create_adjacency_matrix
from square_features import SquareFeatures 

# The model to test
from gnn_sq_model import GNN_sq_BaseModel, GCNLayer

class TestGnnSqModel(unittest.TestCase):

    def test_gcn_layer_forward_pass(self):
        num_nodes = 10
        input_dim = 5
        output_dim = 8
        
        gcn_layer = GCNLayer(input_dim, output_dim)
        
        # Dummy input
        node_features = torch.randn(num_nodes, input_dim)
        adj_matrix = torch.bernoulli(torch.full((num_nodes, num_nodes), 0.3)).float() # Random adj matrix
        
        output = gcn_layer(node_features, adj_matrix)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (num_nodes, output_dim))

    def test_gnn_sq_base_model_forward_pass(self):
        input_dim = SquareFeatures.get_feature_dimension() # Should be 17
        embed_dim = 32 # As defined in GNN_sq_BaseModel default

        model = GNN_sq_BaseModel(input_feature_dim=input_dim, gnn_embedding_dim=embed_dim)

        # Create dummy input similar to what get_gnn_sq_input would produce
        # Node features: (64, 17)
        dummy_node_features_np = np.random.rand(NUM_SQUARES, input_dim).astype(np.float32)
        node_features_tensor = torch.from_numpy(dummy_node_features_np)

        # Adjacency matrix: (64, 64)
        dummy_adj_matrix_np = create_adjacency_matrix(adjacency_type="8_way").astype(np.float32)
        adj_matrix_tensor = torch.from_numpy(dummy_adj_matrix_np)

        # Perform a forward pass
        output_embeddings = model(node_features_tensor, adj_matrix_tensor)

        self.assertIsNotNone(output_embeddings)
        self.assertIsInstance(output_embeddings, torch.Tensor)
        self.assertEqual(output_embeddings.shape, (NUM_SQUARES, embed_dim))
        
        # Check that gradients can be computed (i.e., model is trainable)
        # Requires a dummy loss and backward pass
        if output_embeddings.requires_grad or any(p.requires_grad for p in model.parameters()):
            try:
                # Create a dummy target and loss
                dummy_target = torch.randn(NUM_SQUARES, embed_dim)
                loss = F.mse_loss(output_embeddings, dummy_target)
                
                # Zero gradients, perform a backward pass, and check for grad existence
                model.zero_grad()
                loss.backward()
                
                has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
                self.assertTrue(has_grads, "No gradients were computed during backward pass.")

            except RuntimeError as e:
                self.fail(f"Backward pass failed: {e}")
        else:
            # If no params require grad, this might be an issue or intended (e.g. eval mode with no learnable params)
            # For this model, it should have learnable parameters.
            self.assertTrue(any(p.requires_grad for p in model.parameters()), "Model has no learnable parameters.")


if __name__ == '__main__':
    unittest.main()