import unittest
import torch
from gnn_agent.neural_network.attention_module import CrossAttentionModule

class TestSymmetricCrossAttentionModule(unittest.TestCase):

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

    def test_symmetric_forward_pass_shapes(self):
        """Test the forward pass returns two tensors with correct shapes."""
        num_actual_pieces = 10
        
        square_embeddings = torch.randn(self.num_squares, self.batch_size, self.sq_embed_dim)
        piece_embeddings = torch.randn(num_actual_pieces, self.batch_size, self.pc_embed_dim)
        
        # The module should now return two tensors
        attended_pieces, attended_squares = self.attention_module(square_embeddings, piece_embeddings)
        
        # Verify the piece-centric output shape
        expected_pieces_shape = (num_actual_pieces, self.batch_size, self.pc_embed_dim)
        self.assertEqual(attended_pieces.shape, expected_pieces_shape, "Attended pieces shape is incorrect.")

        # Verify the square-centric output shape
        expected_squares_shape = (self.num_squares, self.batch_size, self.sq_embed_dim)
        self.assertEqual(attended_squares.shape, expected_squares_shape, "Attended squares shape is incorrect.")

    def test_forward_with_padding_mask(self):
        """Test forward pass with a piece padding mask, crucial for S->P attention."""
        num_pieces_per_item = [self.max_num_pieces, self.max_num_pieces - 2, 5, 1]
        
        square_embeddings = torch.randn(self.num_squares, self.batch_size, self.sq_embed_dim)
        piece_embeddings_padded = torch.randn(self.max_num_pieces, self.batch_size, self.pc_embed_dim)
        
        # Create key_padding_mask for the pieces (when they act as Key in S->P)
        key_padding_mask = torch.ones(self.batch_size, self.max_num_pieces, dtype=torch.bool)
        for i, num_p in enumerate(num_pieces_per_item):
            key_padding_mask[i, :num_p] = False # False means "not masked"

        attended_pieces, attended_squares = self.attention_module(
            square_embeddings,
            piece_embeddings_padded,
            piece_padding_mask=key_padding_mask
        )
        
        # Verify the piece-centric output shape (should match the padded input)
        expected_pieces_shape = (self.max_num_pieces, self.batch_size, self.pc_embed_dim)
        self.assertEqual(attended_pieces.shape, expected_pieces_shape, "Padded attended pieces shape is incorrect.")

        # Verify the square-centric output shape
        expected_squares_shape = (self.num_squares, self.batch_size, self.sq_embed_dim)
        self.assertEqual(attended_squares.shape, expected_squares_shape, "Attended squares shape is incorrect with padding.")
        
        self.assertFalse(torch.isnan(attended_squares).any(), "NaNs in square output when using padding mask.")
        self.assertFalse(torch.isnan(attended_pieces).any(), "NaNs in piece output when using padding mask.")


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
        square_embeddings = torch.randn(self.num_squares, self.batch_size, embed_dim)
        piece_embeddings = torch.randn(num_actual_pieces, self.batch_size, embed_dim)
        
        attended_pieces, attended_squares = attention_module_equal_dims(square_embeddings, piece_embeddings)
        
        expected_pieces_shape = (num_actual_pieces, self.batch_size, embed_dim)
        self.assertEqual(attended_pieces.shape, expected_pieces_shape)

        expected_squares_shape = (self.num_squares, self.batch_size, embed_dim)
        self.assertEqual(attended_squares.shape, expected_squares_shape)

    def test_scriptable(self):
        """Test if the new symmetric module can be JIT scripted."""
        try:
            scripted_module = torch.jit.script(self.attention_module)
            
            # Try a forward pass with the scripted module
            num_actual_pieces = 10
            square_embeddings = torch.randn(self.num_squares, self.batch_size, self.sq_embed_dim)
            piece_embeddings = torch.randn(num_actual_pieces, self.batch_size, self.pc_embed_dim)
            
            attended_pieces, attended_squares = scripted_module(square_embeddings, piece_embeddings)
            
            # Verify shapes of both outputs from the scripted module
            self.assertEqual(attended_pieces.shape, (num_actual_pieces, self.batch_size, self.pc_embed_dim))
            self.assertEqual(attended_squares.shape, (self.num_squares, self.batch_size, self.sq_embed_dim))

        except Exception as e:
            self.fail(f"Module scripting failed with symmetric architecture: {e}")

if __name__ == '__main__':
    unittest.main()