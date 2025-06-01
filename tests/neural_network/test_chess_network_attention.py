import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
import os

# Add the project root to the Python path to allow for absolute imports
# This makes the script runnable from anywhere, as long as it's in the project structure
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Correct imports for running tests from the project root directory
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.chess_network import ChessNetwork

# --- Mock GNNs and Heads ---
# These mock classes are simplified to return tensors of expected shapes
# for the purpose of testing the ChessNetwork's attention weight retrieval.

class MockSquareGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, heads):
        super().__init__()
        self.out_features = out_features
        print(f"MockSquareGNN initialized: in={in_features}, hidden={hidden_features}, out={out_features}, heads={heads}")

    def forward(self, features, edge_index):
        # features shape: (num_squares, in_features)
        num_squares = features.size(0)
        # Expected output: (num_squares, out_features)
        return torch.randn(num_squares, self.out_features)

class MockPieceGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        print(f"MockPieceGNN initialized: in={in_channels}, hidden={hidden_channels}, out={out_channels}")

    def forward(self, features, edge_index):
        # features shape: (num_pieces, in_channels)
        num_pieces = features.size(0)
        if num_pieces == 0: # Handle no pieces case
            return torch.empty(0, self.out_channels) 
        # Expected output: (num_pieces, out_channels)
        return torch.randn(num_pieces, self.out_channels)

class MockPolicyHead(nn.Module):
    def __init__(self, embed_dim, num_actions):
        super().__init__()
        self.num_actions = num_actions
        print(f"MockPolicyHead initialized: embed_dim={embed_dim}, num_actions={num_actions}")

    def forward(self, fused_embeddings_batch):
        # fused_embeddings_batch shape: (1, num_squares, embed_dim)
        # Expected output: (1, num_actions)
        batch_size = fused_embeddings_batch.size(0)
        return torch.randn(batch_size, self.num_actions)

class MockValueHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        print(f"MockValueHead initialized: embed_dim={embed_dim}")

    def forward(self, fused_embeddings_batch):
        # fused_embeddings_batch shape: (1, num_squares, embed_dim)
        # Expected output: (1, 1)
        batch_size = fused_embeddings_batch.size(0)
        return torch.randn(batch_size, 1)

# --- End Mock GNNs and Heads ---

def run_test():
    print("--- Starting ChessNetwork Attention Weight Test ---")

    # Network parameters (matching defaults in ChessNetwork)
    square_in_features = 12
    piece_in_features = 12
    embed_dim = 128
    num_actions = 4672
    num_heads_attention = 4 # For CrossAttentionModule
    square_gat_heads = 4    # For SquareGNN
    attention_dropout_rate = 0.1

    # Instantiate the ChessNetwork
    chess_net = ChessNetwork(
        square_in_features=square_in_features,
        piece_in_features=piece_in_features,
        embed_dim=embed_dim,
        num_actions=num_actions,
        num_heads=num_heads_attention,
        square_gat_heads=square_gat_heads,
        attention_dropout_rate=attention_dropout_rate
    )

    # Replace instances with mocks
    # This is a robust way to test the ChessNetwork class in isolation
    chess_net.square_gnn = MockSquareGNN(
        in_features=square_in_features, 
        hidden_features=64,
        out_features=embed_dim, 
        heads=square_gat_heads
    )
    chess_net.piece_gnn = MockPieceGNN(
        in_channels=piece_in_features, 
        hidden_channels=32,
        out_channels=embed_dim
    )
    chess_net.policy_head = MockPolicyHead(embed_dim, num_actions)
    chess_net.value_head = MockValueHead(embed_dim)
    
    # --- Test Case 1: With Pieces ---
    print("\n--- Test Case 1: With Pieces ---")
    num_squares = 64
    num_test_pieces = 10

    # Dummy input tensors
    dummy_square_features = torch.randn(num_squares, square_in_features)
    dummy_square_edge_index = torch.randint(0, num_squares, (2, 100), dtype=torch.long)
    dummy_piece_features = torch.randn(num_test_pieces, piece_in_features)
    dummy_piece_edge_index = torch.randint(0, num_test_pieces, (2, 20), dtype=torch.long)
    dummy_piece_padding_mask = torch.rand(num_test_pieces) > 0.8 # ~20% padded

    print(f"Input shapes:")
    print(f"  Square Features: {dummy_square_features.shape}")
    print(f"  Piece Features: {dummy_piece_features.shape}")
    print(f"  Piece Padding Mask: {dummy_piece_padding_mask.shape}")

    # Forward pass requesting attention weights
    chess_net.eval()
    with torch.no_grad():
        policy_logits, value, attention_weights = chess_net.forward(
            square_features=dummy_square_features,
            square_edge_index=dummy_square_edge_index,
            piece_features=dummy_piece_features,
            piece_edge_index=dummy_piece_edge_index,
            piece_padding_mask=dummy_piece_padding_mask,
            return_attention_weights=True
        )

    print("\nOutput shapes (With Pieces):")
    print(f"  Policy Logits: {policy_logits.shape}")
    print(f"  Value: {value.shape}")
    if attention_weights is not None:
        print(f"  Attention Weights: {attention_weights.shape}")
        assert attention_weights.shape == (num_squares, num_test_pieces)
    else:
        assert False, "Attention weights were None when pieces were present."

    # --- Test Case 2: No Pieces ---
    print("\n--- Test Case 2: No Pieces ---")
    dummy_piece_features_empty = torch.empty(0, piece_in_features)
    dummy_piece_edge_index_empty = torch.empty((2,0), dtype=torch.long)
    dummy_piece_padding_mask_empty = torch.empty(0, dtype=torch.bool)

    print(f"Input shapes (No Pieces):")
    print(f"  Piece Features: {dummy_piece_features_empty.shape}")

    with torch.no_grad():
        policy_logits_empty, value_empty, attention_weights_empty = chess_net.forward(
            square_features=dummy_square_features,
            square_edge_index=dummy_square_edge_index,
            piece_features=dummy_piece_features_empty,
            piece_edge_index=dummy_piece_edge_index_empty,
            piece_padding_mask=dummy_piece_padding_mask_empty,
            return_attention_weights=True
        )

    print("\nOutput shapes (No Pieces):")
    print(f"  Policy Logits: {policy_logits_empty.shape}")
    print(f"  Value: {value_empty.shape}")
    if attention_weights_empty is not None:
        assert False, f"Attention weights were {attention_weights_empty.shape}, expected None."
    else:
        print("  Attention Weights: None (Correct for no-piece case)")
        assert attention_weights_empty is None

    # --- Test Case 3: With Pieces, but no padding mask ---
    print("\n--- Test Case 3: With Pieces, No Padding Mask ---")
    with torch.no_grad():
        policy_logits_no_mask, value_no_mask, attention_weights_no_mask = chess_net.forward(
            square_features=dummy_square_features,
            square_edge_index=dummy_square_edge_index,
            piece_features=dummy_piece_features,
            piece_edge_index=dummy_piece_edge_index,
            piece_padding_mask=None,
            return_attention_weights=True
        )
    
    print("\nOutput shapes (With Pieces, No Padding Mask):")
    print(f"  Policy Logits: {policy_logits_no_mask.shape}")
    print(f"  Value: {value_no_mask.shape}")
    if attention_weights_no_mask is not None:
        print(f"  Attention Weights: {attention_weights_no_mask.shape}")
        assert attention_weights_no_mask.shape == (num_squares, num_test_pieces)
    else:
        assert False, "Attention weights were None (no mask case)."

    print("\n--- ChessNetwork Attention Weight Test Completed Successfully ---")

if __name__ == '__main__':
    run_test()
