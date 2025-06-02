# gnn_agent/neural_network/attention_module.py (Corrected)
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

class CrossAttentionModule(nn.Module):
    def __init__(self, sq_embed_dim: int, pc_embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Cross-Attention Module where piece embeddings attend to square embeddings.

        Args:
            sq_embed_dim (int): Dimension of square embeddings (key/value dimension).
            pc_embed_dim (int): Dimension of piece embeddings (query dimension).
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.sq_embed_dim = sq_embed_dim
        self.pc_embed_dim = pc_embed_dim
        self.num_heads = num_heads

        # MultiHeadAttention layer
        # Query: piece_embeddings (L_pc, N, D_pc)
        # Key: square_embeddings (L_sq, N, D_sq)
        # Value: square_embeddings (L_sq, N, D_sq)
        # embed_dim is the dimension of the query (piece_embeddings)
        # kdim and vdim are the dimensions of the key and value (square_embeddings)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=pc_embed_dim,
            num_heads=num_heads,
            kdim=sq_embed_dim,
            vdim=sq_embed_dim,
            dropout=dropout_rate,
            batch_first=False  # Expected input shape: (seq_len, batch_size, embed_dim)
        )

        # Layer normalization and Feedforward network operate on the query's dimension (pc_embed_dim)
        self.layer_norm1 = nn.LayerNorm(pc_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(pc_embed_dim, pc_embed_dim * 4), # Expansion
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pc_embed_dim * 4, pc_embed_dim)  # Contraction
        )
        self.layer_norm2 = nn.LayerNorm(pc_embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                square_embeddings: torch.Tensor,
                piece_embeddings: torch.Tensor,
                piece_padding_mask: Optional[torch.Tensor] = None, # This mask is for the KEY
                return_attention_weights: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass for the CrossAttentionModule.

        Args:
            square_embeddings (torch.Tensor): The Key/Value. Shape (num_squares, batch_size, sq_embed_dim)
            piece_embeddings (torch.Tensor): The Query. Shape (num_current_pieces, batch_size, pc_embed_dim)
            piece_padding_mask (torch.Tensor, optional): This is not used in piece-to-square attention,
                                                         but a square_padding_mask could be. Kept for API consistency.
            return_attention_weights (bool): If True, returns the attention weights along with the output.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
            - If return_attention_weights is False:
                processed_attended_pieces (torch.Tensor): Attended piece embeddings.
                                                           Shape (num_current_pieces, batch_size, pc_embed_dim)
            - If return_attention_weights is True:
                Tuple containing:
                - processed_attended_pieces (torch.Tensor): As above.
                - attn_output_weights (torch.Tensor): Attention weights.
                                                       Shape (batch_size, num_current_pieces, num_squares).
        """
        # The query should be pieces, the key/value should be squares.
        # This means pieces are "looking at" squares to gather context.
        # query: (L_query, N, E_query) -> (num_pieces, B, D_pc)
        # key: (L_key, N, E_key) -> (num_squares, B, D_sq)
        # value: (L_value, N, E_value) -> (num_squares, B, D_sq)
        
        # attn_output: (L_query, N, E_query) -> (num_pieces, B, D_pc)
        # attn_output_weights: (N, L_query, L_key) -> (B, num_pieces, num_squares)
        
        attn_output, attn_output_weights = self.multi_head_attention(
            query=piece_embeddings,
            key=square_embeddings,
            value=square_embeddings,
            # key_padding_mask is not typically used for the square context, which is never padded.
            need_weights=return_attention_weights
        )
        # If need_weights was False, attn_output_weights will be None.

        # Add & Norm (Residual connection on the query)
        attended_pieces = self.layer_norm1(piece_embeddings + self.dropout(attn_output))

        # Feedforward
        ff_output = self.feed_forward(attended_pieces)

        # Add & Norm (Residual connection)
        processed_attended_pieces = self.layer_norm2(attended_pieces + self.dropout(ff_output))

        if return_attention_weights:
            return processed_attended_pieces, attn_output_weights
        else:
            return processed_attended_pieces