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
            embed_dim=pc_embed_dim,
            num_heads=num_heads,
            kdim=sq_embed_dim,
            vdim=sq_embed_dim,
            dropout=dropout_rate,
            batch_first=False
        )

        # Attention mechanism 2: Squares attend to Pieces (S -> P)
        self.s_to_p_attention = nn.MultiheadAttention(
            embed_dim=sq_embed_dim,
            num_heads=num_heads,
            kdim=pc_embed_dim,
            vdim=pc_embed_dim,
            dropout=dropout_rate,
            batch_first=False
        )

        # Processing block for the P -> S path
        self.p_layer_norm1 = nn.LayerNorm(pc_embed_dim)
        self.p_feed_forward = nn.Sequential(
            nn.Linear(pc_embed_dim, pc_embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pc_embed_dim * 4, pc_embed_dim)
        )
        self.p_layer_norm2 = nn.LayerNorm(pc_embed_dim)
        self.p_dropout = nn.Dropout(dropout_rate)

        # Processing block for the S -> P path
        self.s_layer_norm1 = nn.LayerNorm(sq_embed_dim)
        self.s_feed_forward = nn.Sequential(
            nn.Linear(sq_embed_dim, sq_embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(sq_embed_dim * 4, sq_embed_dim)
        )
        self.s_layer_norm2 = nn.LayerNorm(sq_embed_dim)
        self.s_dropout = nn.Dropout(dropout_rate)

    def forward(self,
                square_embeddings: torch.Tensor,
                piece_embeddings: torch.Tensor,
                piece_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the SymmetricCrossAttentionModule.

        Args:
            square_embeddings (torch.Tensor): Shape (num_squares, batch_size, sq_embed_dim)
            piece_embeddings (torch.Tensor): Shape (num_current_pieces, batch_size, pc_embed_dim)
            piece_padding_mask (torch.Tensor, optional): Mask for padded pieces. 
                                                         Shape (batch_size, num_current_pieces).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - processed_attended_pieces (torch.Tensor):
                Piece embeddings after attending to squares.
                Shape (num_current_pieces, batch_size, pc_embed_dim).
            - processed_attended_squares (torch.Tensor):
                Square embeddings after attending to pieces.
                Shape (num_squares, batch_size, sq_embed_dim).
        """
        # --- Path 1: Pieces attend to Squares (P -> S) ---
        p_to_s_attn_output, _ = self.p_to_s_attention(
            query=piece_embeddings,
            key=square_embeddings,
            value=square_embeddings,
            need_weights=False
        )
        attended_pieces = self.p_layer_norm1(piece_embeddings + self.p_dropout(p_to_s_attn_output))
        p_ff_output = self.p_feed_forward(attended_pieces)
        processed_attended_pieces = self.p_layer_norm2(attended_pieces + self.p_dropout(p_ff_output))

        # --- Path 2: Squares attend to Pieces (S -> P) ---
        s_to_p_attn_output, _ = self.s_to_p_attention(
            query=square_embeddings,
            key=piece_embeddings,
            value=piece_embeddings,
            key_padding_mask=piece_padding_mask, # Crucial for ignoring padded pieces
            need_weights=False
        )
        attended_squares = self.s_layer_norm1(square_embeddings + self.s_dropout(s_to_p_attn_output))
        s_ff_output = self.s_feed_forward(attended_squares)
        processed_attended_squares = self.s_layer_norm2(attended_squares + self.s_dropout(s_ff_output))

        return processed_attended_pieces, processed_attended_squares