# gnn_agent/neural_network/chess_network.py (Corrected)

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from .gnn_models import SquareGNN
from .gnn_models import PieceGNN
from .attention_module import CrossAttentionModule
from .policy_value_heads import PolicyHead
from .policy_value_heads import ValueHead

class ChessNetwork(nn.Module):
    """
    The main neural network that integrates the dual GNNs, cross-attention,
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

        # This layer was in the original plan but might not be needed if
        # the heads can handle the fused embedding dimension directly.
        # Let's assume the heads take the raw concatenated dimension.
        # For now, we will bypass this extra layer as it might complicate things.
        # embed_dim = cross_attention.sq_embed_dim
        # self.embedding_layer = nn.Linear(embed_dim * 2, embed_dim)


    def forward(self,
                    square_features: torch.Tensor,
                    square_edge_index: torch.Tensor,
                    piece_features: torch.Tensor,
                    piece_edge_index: torch.Tensor,
                    piece_to_square_map: Optional[torch.Tensor] = None,
                    return_attention: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor],
                            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
            """
            Forward pass for the full network.
            """
            square_embeddings = self.square_gnn(square_features, square_edge_index)
            piece_embeddings = self.piece_gnn(piece_features, piece_edge_index)

            batched_attention_weights = None
            
            final_square_representation = square_embeddings

            if piece_embeddings.numel() > 0 and piece_to_square_map is not None:
                # CORRECTED: Use unsqueeze(1) for batch_first=False attention modules.
                # This creates the shape (seq_len, batch_size=1, embed_dim).
                query_b = piece_embeddings.unsqueeze(1)
                key_b = square_embeddings.unsqueeze(1)
                value_b = square_embeddings.unsqueeze(1)

                attention_output = self.cross_attention(
                    square_embeddings=key_b,
                    piece_embeddings=query_b,
                    return_attention_weights=return_attention
                )
                
                if return_attention:
                    attended_piece_embeddings_b, weights_b = attention_output
                    if weights_b is not None:
                        batched_attention_weights = weights_b.squeeze(0)
                else:
                    attended_piece_embeddings_b = attention_output

                # Squeeze the batch dimension after attention.
                attended_piece_embeddings = attended_piece_embeddings_b.squeeze(1)

                final_square_representation = torch.zeros_like(square_embeddings)
                final_square_representation.index_add_(0, piece_to_square_map, attended_piece_embeddings)

            # Add a batch dimension for the policy and value heads.
            final_representation_for_heads = final_square_representation.unsqueeze(0)

            policy_logits_b = self.policy_head(final_representation_for_heads)
            value_b = self.value_head(final_representation_for_heads)

            # Remove batch dimension for the final output.
            policy_logits = policy_logits_b.squeeze(0)
            final_value = value_b.squeeze(0)

            if return_attention:
                return policy_logits, final_value, batched_attention_weights
            else:
                return policy_logits, final_value