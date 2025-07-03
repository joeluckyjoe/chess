import torch
import torch.nn as nn
from typing import Optional, Tuple

class CrossAttentionModule(nn.Module):
    def __init__(self, sq_embed_dim: int, pc_embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Symmetric Cross-Attention Module with two-way attention:
        1. Piece embeddings attend to Square embeddings (P -> S).
        2. Square embeddings attend to Piece embeddings (S -> P).

        Args:
            sq_embed_dim (int): Dimension of square embeddings.
            pc_embed_dim (int): Dimension of piece embeddings.
            num_heads (int): Number of attention heads for both attention mechanisms.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.sq_embed_dim = sq_embed_dim
        self.pc_embed_dim = pc_embed_dim
        self.num_heads = num_heads

        # Attention mechanism 1: Pieces attend to Squares (P -> S)
        self.p_to_s_attention = nn.MultiheadAttention(
            embed_dim=pc_embed_dim,  # Query dim
            num_heads=num_heads,
            kdim=sq_embed_dim,       # Key dim
            vdim=sq_embed_dim,       # Value dim
            dropout=dropout_rate,
            batch_first=True
        )

        # Attention mechanism 2: Squares attend to Pieces (S -> P)
        self.s_to_p_attention = nn.MultiheadAttention(
            embed_dim=sq_embed_dim,  # Query dim
            num_heads=num_heads,
            kdim=pc_embed_dim,       # Key dim
            vdim=pc_embed_dim,       # Value dim
            dropout=dropout_rate,
            batch_first=True
        )

        # --- FIX: Add projection layers to match dimensions for residual connections ---
        self.p_projection = nn.Linear(sq_embed_dim, pc_embed_dim)
        self.s_projection = nn.Linear(pc_embed_dim, sq_embed_dim)
        # --- END FIX ---

        # Processing block for the P -> S path
        self.p_layer_norm1 = nn.LayerNorm(pc_embed_dim)
        self.p_feed_forward = nn.Sequential(
            nn.Linear(pc_embed_dim, pc_embed_dim * 4),
            nn.GELU(), # Replaced ReLU with GELU as per previous findings
            nn.Dropout(dropout_rate),
            nn.Linear(pc_embed_dim * 4, pc_embed_dim)
        )
        self.p_layer_norm2 = nn.LayerNorm(pc_embed_dim)
        self.p_dropout = nn.Dropout(dropout_rate)

        # Processing block for the S -> P path
        self.s_layer_norm1 = nn.LayerNorm(sq_embed_dim)
        self.s_feed_forward = nn.Sequential(
            nn.Linear(sq_embed_dim, sq_embed_dim * 4),
            nn.GELU(), # Replaced ReLU with GELU as per previous findings
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
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for the SymmetricCrossAttentionModule.

        Args:
            square_embeddings (torch.Tensor): Shape (batch_size, num_squares, sq_embed_dim)
            piece_embeddings (torch.Tensor): Shape (batch_size, num_current_pieces, pc_embed_dim)
            piece_padding_mask (torch.Tensor, optional): Mask for padded pieces.
                                                          Shape (batch_size, num_current_pieces).
            return_attention (bool): If True, returns the attention weight tensors.

        Returns:
            Tuple: A tuple containing:
            - processed_attended_pieces (torch.Tensor)
            - processed_attended_squares (torch.Tensor)
            - ps_weights (torch.Tensor or None): Piece-to-Square attention weights.
            - sp_weights (torch.Tensor or None): Square-to-Piece attention weights.
        """
        # --- Path 1: Pieces attend to Squares (P -> S) ---
        p_to_s_attn_output, ps_weights = self.p_to_s_attention(
            query=piece_embeddings,
            key=square_embeddings,
            value=square_embeddings,
            need_weights=return_attention
        )
        # --- FIX: Project attention output back to piece_embedding dimension ---
        projected_p_output = self.p_projection(p_to_s_attn_output)
        # --- END FIX ---
        attended_pieces = self.p_layer_norm1(piece_embeddings + self.p_dropout(projected_p_output))
        p_ff_output = self.p_feed_forward(attended_pieces)
        processed_attended_pieces = self.p_layer_norm2(attended_pieces + self.p_dropout(p_ff_output))

        # --- Path 2: Squares attend to Pieces (S -> P) ---
        s_to_p_attn_output, sp_weights = self.s_to_p_attention(
            query=square_embeddings,
            key=piece_embeddings,
            value=piece_embeddings,
            key_padding_mask=piece_padding_mask,
            need_weights=return_attention
        )
        # --- FIX: Project attention output back to square_embedding dimension ---
        projected_s_output = self.s_projection(s_to_p_attn_output)
        # --- END FIX ---
        attended_squares = self.s_layer_norm1(square_embeddings + self.s_dropout(projected_s_output))
        s_ff_output = self.s_feed_forward(attended_squares)
        processed_attended_squares = self.s_layer_norm2(attended_squares + self.s_dropout(s_ff_output))

        return processed_attended_pieces, processed_attended_squares, ps_weights, sp_weights