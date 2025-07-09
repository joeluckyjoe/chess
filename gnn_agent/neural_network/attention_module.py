import torch
import torch.nn as nn
from typing import Optional, Tuple

class CrossAttentionModule(nn.Module):
    def __init__(self, sq_embed_dim: int, pc_embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Asymmetric Cross-Attention Module to test the Architectural Regression Hypothesis (Phase AY).
        This module implements a one-way attention flow:
        1. Square embeddings attend to Piece embeddings (S -> P) to create a final board context.
        The P -> S path has been disabled.

        Args:
            sq_embed_dim (int): Dimension of square embeddings.
            pc_embed_dim (int): Dimension of piece embeddings.
            num_heads (int): Number of attention heads for the attention mechanism.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.sq_embed_dim = sq_embed_dim
        self.pc_embed_dim = pc_embed_dim
        self.num_heads = num_heads

        # Attention mechanism: Squares attend to Pieces (S -> P)
        self.s_to_p_attention = nn.MultiheadAttention(
            embed_dim=sq_embed_dim,    # Query dim (Squares)
            num_heads=num_heads,
            kdim=pc_embed_dim,       # Key dim (Pieces)
            vdim=pc_embed_dim,       # Value dim (Pieces)
            dropout=dropout_rate,
            batch_first=True
        )

        # --- FIX ---
        # The projection layer was removed. The output of MultiheadAttention (s_to_p_attn_output)
        # already has the correct dimension (sq_embed_dim) to be added to the residual
        # connection (square_embeddings). The original projection layer was both unnecessary
        # and incorrectly defined, causing the unit test failures.
        # self.s_projection = nn.Linear(pc_embed_dim, sq_embed_dim) # This was the buggy line.

        # Processing block for the S -> P path
        self.s_layer_norm1 = nn.LayerNorm(sq_embed_dim)
        self.s_feed_forward = nn.Sequential(
            nn.Linear(sq_embed_dim, sq_embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(sq_embed_dim * 4, sq_embed_dim)
        )
        self.s_layer_norm2 = nn.LayerNorm(sq_embed_dim)
        self.s_dropout = nn.Dropout(dropout_rate)

    def forward(self,
                square_embeddings: torch.Tensor,
                piece_embeddings: torch.Tensor,
                piece_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Asymmetric CrossAttentionModule.

        Args:
            square_embeddings (torch.Tensor): Shape (batch_size, num_squares, sq_embed_dim)
            piece_embeddings (torch.Tensor): Shape (batch_size, num_current_pieces, pc_embed_dim)
            piece_padding_mask (torch.Tensor, optional): Mask for padded pieces.
                                                          Shape (batch_size, num_current_pieces).
            return_attention (bool): If True, returns the attention weight tensor.

        Returns:
            Tuple: A tuple containing:
            - processed_attended_squares (torch.Tensor): The updated square embeddings.
            - sp_weights (torch.Tensor or None): Square-to-Piece attention weights.
        """
        # --- Path: Squares attend to Pieces (S -> P) ---
        s_to_p_attn_output, sp_weights = self.s_to_p_attention(
            query=square_embeddings,
            key=piece_embeddings,
            value=piece_embeddings,
            key_padding_mask=piece_padding_mask,
            need_weights=return_attention
        )

        # First residual connection and normalization, using the attention output directly.
        attended_squares = self.s_layer_norm1(square_embeddings + self.s_dropout(s_to_p_attn_output))

        # Feed-forward network
        s_ff_output = self.s_feed_forward(attended_squares)

        # Second residual connection and normalization
        processed_attended_squares = self.s_layer_norm2(attended_squares + self.s_dropout(s_ff_output))

        return processed_attended_squares, sp_weights