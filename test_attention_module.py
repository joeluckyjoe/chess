# test_attention_module.py
import unittest
import torch
from attention_module import CrossAttentionModule # Assuming the file is named attention_module.py

class TestCrossAttentionModule(unittest.TestCase):

    def setUp(self):
        self.sq_embed_dim = 128
        self.pc_embed_dim = 96
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.batch_size = 4
        self.num_squares = 64 # Fixed for chess
        self.max_num_pieces = 16 # Max pieces one side can have, for padding example (can vary)

        self.attention_module = CrossAttentionModule(
            sq_embed_dim=self.sq_embed_dim,
            pc_embed_dim=self.pc_embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        self.attention_module.eval() # Set to evaluation mode for consistent dropout behavior

    def test_forward_pass_basic_shapes(self):
        """Test forward pass with basic shapes and no padding."""
        num_actual_pieces = 10 # Variable number of pieces for this test case
        
        # Create dummy input tensors
        # Shapes: (seq_len, batch_size, embed_dim)
        square_embeddings = torch.randn(self.num_squares, self.batch_size, self.sq_embed_dim)
        piece_embeddings = torch.randn(num_actual_pieces, self.batch_size, self.pc_embed_dim)
        
        output = self.attention_module(square_embeddings, piece_embeddings)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.num_squares, self.batch_size, self.sq_embed_dim))

    def test_forward_pass_with_padding_mask(self):
        """Test forward pass with a key padding mask."""
        # Simulate a scenario where piece tensors are padded to max_num_pieces
        # For this test, let's say some items in the batch have fewer than max_num_pieces
        num_pieces_per_item = [self.max_num_pieces, self.max_num_pieces - 2, self.max_num_pieces // 2, 5]
        assert self.batch_size == len(num_pieces_per_item), "Batch size for test doesn't match num_pieces_per_item length"

        # (seq_len, batch_size, embed_dim)
        square_embeddings = torch.randn(self.num_squares, self.batch_size, self.sq_embed_dim)
        # piece_embeddings are padded to max_num_pieces along the sequence dimension
        piece_embeddings_padded = torch.randn(self.max_num_pieces, self.batch_size, self.pc_embed_dim)
        
        # Create key_padding_mask (batch_size, seq_len_key)
        # True means the key position will be ignored.
        key_padding_mask = torch.ones(self.batch_size, self.max_num_pieces, dtype=torch.bool)
        for i, num_p in enumerate(num_pieces_per_item):
            if num_p > 0 : # Ensure no error if num_p is 0, though typically not expected if pieces exist
                 key_padding_mask[i, :num_p] = False # These are the actual pieces, not masked

        output = self.attention_module(square_embeddings, piece_embeddings_padded, piece_padding_mask=key_padding_mask)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.num_squares, self.batch_size, self.sq_embed_dim))

    def test_forward_pass_no_pieces(self):
        """Test forward pass when there are no pieces (e.g., empty board or error state)."""
        # (seq_len, batch_size, embed_dim)
        square_embeddings = torch.randn(self.num_squares, self.batch_size, self.sq_embed_dim)
        
        # Case 1: piece_embeddings tensor has 0 pieces (seq_len = 0)
        # This is tricky for MHA. It might error or produce NaNs if seq_len is 0 for key/value.
        # nn.MultiheadAttention typically expects L_source > 0.
        # Let's test with a very small number of pieces instead of zero, or handle it.
        # For this test, we'll assume the calling code ensures at least one "dummy" piece if needed,
        # or that MHA handles it gracefully if `key_padding_mask` masks all.
        
        # Let's test with all pieces masked.
        num_actual_pieces = self.max_num_pieces # or some positive number
        piece_embeddings_all_padded = torch.randn(num_actual_pieces, self.batch_size, self.pc_embed_dim)
        key_padding_mask_all_true = torch.ones(self.batch_size, num_actual_pieces, dtype=torch.bool)

        output = self.attention_module(square_embeddings, piece_embeddings_all_padded, piece_padding_mask=key_padding_mask_all_true)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.num_squares, self.batch_size, self.sq_embed_dim))
        # We should also check that output is not NaN, indicating a potential issue with all-masked attention.
        self.assertFalse(torch.isnan(output).any(), "Output contains NaNs when all pieces are masked.")


    def test_scriptable(self):
        """Test if the module can be JIT scripted."""
        try:
            scripted_module = torch.jit.script(self.attention_module)
            # Try a forward pass with scripted module
            num_actual_pieces = 10
            square_embeddings = torch.randn(self.num_squares, self.batch_size, self.sq_embed_dim)
            piece_embeddings = torch.randn(num_actual_pieces, self.batch_size, self.pc_embed_dim)
            output = scripted_module(square_embeddings, piece_embeddings)
            self.assertEqual(output.shape, (self.num_squares, self.batch_size, self.sq_embed_dim))
        except Exception as e:
            self.fail(f"Module scripting failed: {e}")

if __name__ == '__main__':
    unittest.main()