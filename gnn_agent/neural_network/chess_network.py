# gnn_agent/neural_network/chess_network.py

import torch
import torch.nn as nn

from .gnn_models import SquareGNN, PieceGNN
from .attention_module import CrossAttentionModule
from .policy_value_heads import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main neural network that integrates the dual GNNs, cross-attention,
    and policy/value heads.
    """
    def __init__(self,
                 # SquareGNN params
                 square_in_features: int = 14,
                 square_hidden_features: int = 64,
                 square_out_features: int = 128,
                 square_gat_heads: int = 4,
                 # PieceGNN params
                 piece_in_features: int = 6,
                 piece_hidden_features: int = 32,
                 piece_out_features: int = 128,
                 # Attention params
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 # Heads params
                 board_size: int = 64,
                 num_actions: int = 4672):
        super(ChessNetwork, self).__init__()

        # Dual GNN encoders
        self.square_gnn = SquareGNN(
            in_features=square_in_features,
            hidden_features=square_hidden_features,
            out_features=embed_dim,
            heads=square_gat_heads
        )
        self.piece_gnn = PieceGNN(
            in_channels=piece_in_features,
            hidden_channels=piece_hidden_features,
            out_channels=embed_dim
        )

        # Cross-attention module to fuse GNN outputs
        self.cross_attention = CrossAttentionModule(embed_dim, num_heads)

        # Combined embedding layer
        self.embedding_layer = nn.Linear(embed_dim * 2, embed_dim) # From concatenation

        # Policy and Value Heads
        self.policy_head = PolicyHead(embed_dim, board_size, num_actions)
        self.value_head = ValueHead(embed_dim, board_size)

    def forward(self, square_features, square_edge_index, piece_features, piece_edge_index, piece_to_square_map):
        """
        Forward pass for the full network.

        Args:
            square_features (Tensor): Node features for all 64 squares.
            square_edge_index (Tensor): Edge index for the square graph.
            piece_features (Tensor): Node features for the pieces on the board.
            piece_edge_index (Tensor): Edge index for the piece graph.
            piece_to_square_map (Tensor): Maps piece embedding indices to their square indices.

        Returns:
            policy_logits (Tensor): Raw output for the policy head.
            value (Tensor): Scalar value prediction for the current board state.
        """
        # 1. Process inputs through GNNs
        square_embeddings = self.square_gnn(square_features, square_edge_index)
        piece_embeddings = self.piece_gnn(piece_features, piece_edge_index)

        # Handle case where there are no pieces
        if piece_embeddings.size(0) == 0:
             # If no pieces, the state is primarily defined by square control
             fused_embeddings = square_embeddings
        else:
            # 2. Apply cross-attention
            # Use square embeddings as query, piece embeddings as key/value
            attended_piece_embeddings = self.cross_attention(
                query=square_embeddings,
                key=piece_embeddings,
                value=piece_embeddings
            )

            # 3. Combine embeddings
            # Concatenate the original square embeddings with the attended piece info
            combined = torch.cat([square_embeddings, attended_piece_embeddings], dim=1)
            fused_embeddings = self.embedding_layer(combined)

        # 4. Pass fused embeddings to heads
        policy_logits = self.policy_head(fused_embeddings)
        value = self.value_head(fused_embeddings)

        return policy_logits, torch.tanh(value)