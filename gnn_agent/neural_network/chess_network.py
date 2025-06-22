import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from .gnn_models import SquareGNN
from .gnn_models import PieceGNN
from .attention_module import CrossAttentionModule
from .policy_value_heads import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main neural network that integrates the dual GNNs, symmetric cross-attention,
    and policy/value heads. This class accepts pre-initialized components
    for modularity and testability.
    """
    def __init__(self,
                 square_gnn: SquareGNN,
                 piece_gnn: PieceGNN,
                 cross_attention: CrossAttentionModule,
                 policy_head: PolicyHead,
                 value_head: ValueHead):
        """
        Initializes the ChessNetwork by accepting its pre-initialized components.
        """
        super(ChessNetwork, self).__init__()

        self.square_gnn = square_gnn
        self.piece_gnn = piece_gnn
        self.cross_attention = cross_attention
        self.policy_head = policy_head
        self.value_head = value_head

        self.embedding_layer = nn.Linear(
            cross_attention.pc_embed_dim + cross_attention.sq_embed_dim,
            cross_attention.sq_embed_dim
        )
        self.relu = nn.ReLU()


    def forward(self,
                square_features: torch.Tensor,
                square_edge_index: torch.Tensor,
                piece_features: torch.Tensor,
                piece_edge_index: torch.Tensor,
                piece_to_square_map: torch.Tensor,
                piece_padding_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False
               ) -> Union[Tuple[torch.Tensor, torch.Tensor],
                          Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Forward pass for the full network using symmetric cross-attention.
        """
        square_embeddings = self.square_gnn(square_features, square_edge_index)
        piece_embeddings = self.piece_gnn(piece_features, piece_edge_index)

        final_square_representation = square_embeddings
        
        ps_attn_weights, sp_attn_weights = None, None

        if piece_embeddings.numel() > 0:
            sq_embed_b = square_embeddings.unsqueeze(1)
            pc_embed_b = piece_embeddings.unsqueeze(1)

            attended_piece_embeddings_b, attended_square_embeddings_b, ps_attn_weights_b, sp_attn_weights_b = self.cross_attention(
                square_embeddings=sq_embed_b,
                piece_embeddings=pc_embed_b,
                piece_padding_mask=piece_padding_mask,
                return_attention=return_attention
            )

            attended_piece_embeddings = attended_piece_embeddings_b.squeeze(1)
            attended_square_embeddings = attended_square_embeddings_b.squeeze(1)

            if return_attention and ps_attn_weights_b is not None:
                # --- THIS IS THE FIX ---
                # The batch dimension for attention weights is 0. Changed squeeze(1) to squeeze(0).
                ps_attn_weights = ps_attn_weights_b.squeeze(0)
                sp_attn_weights = sp_attn_weights_b.squeeze(0)

            piece_centric_board_representation = torch.zeros_like(square_embeddings)
            if attended_piece_embeddings.shape[1] != piece_centric_board_representation.shape[1]:
                 raise ValueError("Feature dimensions of piece and square embeddings must match for index_add_.")

            piece_centric_board_representation.index_add_(0, piece_to_square_map, attended_piece_embeddings)

            fused_representation = torch.cat(
                [piece_centric_board_representation, attended_square_embeddings],
                dim=1
            )

            final_square_representation = self.relu(self.embedding_layer(fused_representation))

        final_representation_for_heads = final_square_representation.unsqueeze(0)

        policy_logits_b = self.policy_head(final_representation_for_heads)
        value_b = self.value_head(final_representation_for_heads)

        policy_logits = policy_logits_b.squeeze(0)
        final_value = value_b.squeeze(0)
        
        if return_attention:
            return policy_logits, final_value, ps_attn_weights, sp_attn_weights
        else:
            return policy_logits, final_value