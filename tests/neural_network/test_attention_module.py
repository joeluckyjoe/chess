import unittest
import torch
from gnn_agent.neural_network.attention_module import CrossAttentionModule

class TestAsymmetricCrossAttentionModule(unittest.TestCase):
    """
    Unit tests for the AsymmetricCrossAttentionModule, updated for Phase AY.
    These tests validate the one-way (S->P) attention flow and correct tensor shapes,
    especially with batch_first=True.
    """

    def setUp(self):
        """Set up a standard configuration for the attention module tests."""
        self.sq_embed_dim = 128
        self.pc_embed_dim = 96
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.batch_size = 4
        self.num_squares = 64  # Fixed for chess
        self.max_num_pieces = 16 # Max pieces for padding

        self.attention_module = CrossAttentionModule(
            sq_embed_dim=self.sq_embed_dim,
            pc_embed_dim=self.pc_embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        self.attention_module.eval() # Set to evaluation mode for consistent dropout

    def test_asymmetric_forward_pass_shapes(self):
        """Test the forward pass returns one tensor with the correct shape."""
        num_actual_pieces = 10
        
        # Corrected shape for batch_first=True: (batch, seq, feature)
        square_embeddings = torch.randn(self.batch_size, self.num_squares, self.sq_embed_dim)
        piece_embeddings = torch.randn(self.batch_size, num_actual_pieces, self.pc_embed_dim)
        
        # The module should now return one primary tensor (attended_squares) and optional weights
        attended_squares, _ = self.attention_module(square_embeddings, piece_embeddings)
        
        # Verify the square-centric output shape
        expected_squares_shape = (self.batch_size, self.num_squares, self.sq_embed_dim)
        self.assertEqual(attended_squares.shape, expected_squares_shape, "Attended squares shape is incorrect.")

    def test_forward_with_padding_mask(self):
        """Test forward pass with a piece padding mask, crucial for S->P attention."""
        num_pieces_per_item = [self.max_num_pieces, self.max_num_pieces - 2, 5, 1]
        
        # Corrected shape for batch_first=True
        square_embeddings = torch.randn(self.batch_size, self.num_squares, self.sq_embed_dim)
        piece_embeddings_padded = torch.randn(self.batch_size, self.max_num_pieces, self.pc_embed_dim)
        
        # Create key_padding_mask for the pieces (when they act as Key in S->P)
        key_padding_mask = torch.ones(self.batch_size, self.max_num_pieces, dtype=torch.bool)
        for i, num_p in enumerate(num_pieces_per_item):
            key_padding_mask[i, :num_p] = False # False means "not masked"

        attended_squares, _ = self.attention_module(
            square_embeddings,
            piece_embeddings_padded,
            piece_padding_mask=key_padding_mask
        )
        
        # Verify the square-centric output shape
        expected_squares_shape = (self.batch_size, self.num_squares, self.sq_embed_dim)
        self.assertEqual(attended_squares.shape, expected_squares_shape, "Attended squares shape is incorrect with padding.")
        
        self.assertFalse(torch.isnan(attended_squares).any(), "NaNs in square output when using padding mask.")

    def test_forward_pass_equal_dims(self):
        """Test the forward pass when square and piece embedding dimensions are equal."""
        embed_dim = 128
        attention_module_equal_dims = CrossAttentionModule(
            sq_embed_dim=embed_dim,
            pc_embed_dim=embed_dim,
            num_heads=self.num_heads
        )
        attention_module_equal_dims.eval()
        
        num_actual_pieces = 12
        # Corrected shape for batch_first=True
        square_embeddings = torch.randn(self.batch_size, self.num_squares, embed_dim)
        piece_embeddings = torch.randn(self.batch_size, num_actual_pieces, embed_dim)
        
        attended_squares, _ = attention_module_equal_dims(square_embeddings, piece_embeddings)
        
        expected_squares_shape = (self.batch_size, self.num_squares, embed_dim)
        self.assertEqual(attended_squares.shape, expected_squares_shape)

    def test_return_attention_weights(self):
        """Test that attention weights are returned with the correct shape when requested."""
        num_actual_pieces = 12
        
        # Corrected shape for batch_first=True
        square_embeddings = torch.randn(self.batch_size, self.num_squares, self.sq_embed_dim)
        piece_embeddings = torch.randn(self.batch_size, num_actual_pieces, self.pc_embed_dim)

        _, sp_weights = self.attention_module(
            square_embeddings,
            piece_embeddings,
            return_attention=True
        )

        self.assertIsNotNone(sp_weights, "Attention weights should not be None when return_attention=True.")
        
        # Expected shape: (batch_size, num_heads, num_queries, num_keys)
        # However, MultiheadAttention averages over heads, so shape is (batch_size, num_queries, num_keys)
        # num_queries = num_squares, num_keys = num_actual_pieces
        expected_weights_shape = (self.batch_size, self.num_squares, num_actual_pieces)
        self.assertEqual(sp_weights.shape, expected_weights_shape, "S->P attention weights have an incorrect shape.")

    def test_scriptable(self):
        """Test if the new asymmetric module can be JIT scripted."""
        try:
            scripted_module = torch.jit.script(self.attention_module)
            
            # Try a forward pass with the scripted module
            num_actual_pieces = 10
            # Corrected shape for batch_first=True
            square_embeddings = torch.randn(self.batch_size, self.num_squares, self.sq_embed_dim)
            piece_embeddings = torch.randn(self.batch_size, num_actual_pieces, self.pc_embed_dim)
            
            attended_squares, _ = scripted_module(square_embeddings, piece_embeddings)
            
            # Verify shape of the output from the scripted module
            self.assertEqual(attended_squares.shape, (self.batch_size, self.num_squares, self.sq_embed_dim))

        except Exception as e:
            self.fail(f"Module scripting failed with asymmetric architecture: {e}")

if __name__ == '__main__':
    unittest.main()
