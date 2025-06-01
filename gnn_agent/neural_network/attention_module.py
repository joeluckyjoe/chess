# attention_module.py
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

class CrossAttentionModule(nn.Module):
    def __init__(self, sq_embed_dim: int, pc_embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Cross-Attention Module where square embeddings attend to piece embeddings.

        Args:
            sq_embed_dim (int): Dimension of square embeddings (query dimension).
            pc_embed_dim (int): Dimension of piece embeddings (key/value dimension).
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.sq_embed_dim = sq_embed_dim
        self.pc_embed_dim = pc_embed_dim
        self.num_heads = num_heads

        # MultiHeadAttention layer
        # Query: square_embeddings (L_sq, N, D_sq)
        # Key: piece_embeddings (L_pc, N, D_pc)
        # Value: piece_embeddings (L_pc, N, D_pc)
        # embed_dim is the dimension of the query (square_embeddings)
        # kdim and vdim are the dimensions of the key and value (piece_embeddings)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=sq_embed_dim,
            num_heads=num_heads,
            kdim=pc_embed_dim,
            vdim=pc_embed_dim,
            dropout=dropout_rate,
            batch_first=False  # Expected input shape: (seq_len, batch_size, embed_dim)
        )

        # Layer normalization and Feedforward network (optional, but common)
        self.layer_norm1 = nn.LayerNorm(sq_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(sq_embed_dim, sq_embed_dim * 4), # Expansion
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(sq_embed_dim * 4, sq_embed_dim)  # Contraction
        )
        self.layer_norm2 = nn.LayerNorm(sq_embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                square_embeddings: torch.Tensor,
                piece_embeddings: torch.Tensor,
                piece_padding_mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass for the CrossAttentionModule.

        Args:
            square_embeddings (torch.Tensor): Shape (num_squares, batch_size, sq_embed_dim)
            piece_embeddings (torch.Tensor): Shape (num_current_pieces, batch_size, pc_embed_dim)
            piece_padding_mask (torch.Tensor, optional): Boolean tensor for key padding.
                                                         Shape (batch_size, num_current_pieces).
                                                         True indicates a position to be masked.
                                                         Defaults to None.
            return_attention_weights (bool): If True, returns the attention weights along with the output.
                                             Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
            - If return_attention_weights is False:
                processed_attended_squares (torch.Tensor): Attended square embeddings.
                                                            Shape (num_squares, batch_size, sq_embed_dim)
            - If return_attention_weights is True:
                Tuple containing:
                - processed_attended_squares (torch.Tensor): As above.
                - attn_output_weights (torch.Tensor): Attention weights.
                                                      Shape (batch_size, num_squares, num_current_pieces).
                                                      Will be None if not computed.
        """
        # square_embeddings: (L_target, N, E_query) -> (64, B, D_sq)
        # piece_embeddings: (L_source, N, E_kv) -> (N_pieces, B, D_pc)

        # Apply multi-head attention
        # attn_output: (L_target, N, E_query)
        # attn_output_weights: (N, L_target, L_source) if need_weights=True
        #                      (batch_size, num_squares, num_current_pieces)
        
        # Determine if weights are needed for the underlying call
        # This avoids computing them if not requested by the caller of this forward method
        _need_weights = return_attention_weights 

        attn_output, attn_output_weights = self.multi_head_attention(
            query=square_embeddings,
            key=piece_embeddings,
            value=piece_embeddings,
            key_padding_mask=piece_padding_mask, # (N, S_key)
            need_weights=_need_weights # Only compute if explicitly asked
        )
        # If _need_weights was False, attn_output_weights will be None.

        # Add & Norm (Residual connection)
        attended_squares = self.layer_norm1(square_embeddings + self.dropout(attn_output))

        # Feedforward
        ff_output = self.feed_forward(attended_squares)

        # Add & Norm (Residual connection)
        processed_attended_squares = self.layer_norm2(attended_squares + self.dropout(ff_output))

        if return_attention_weights:
            return processed_attended_squares, attn_output_weights
        else:
            return processed_attended_squares
