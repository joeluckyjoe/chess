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
        attended_square_embeddings = torch.zeros_like(square_embeddings)

        if piece_embeddings.numel() > 0:
            # Add a batch dimension of 1 for the attention module
            square_embeddings_b = square_embeddings.unsqueeze(1)
            piece_embeddings_b = piece_embeddings.unsqueeze(1)

            # --- CORRECTED LOGIC ---
            # Call the attention module and handle its output based on the flag.
            attention_output = self.cross_attention(
                square_embeddings=square_embeddings_b,
                piece_embeddings=piece_embeddings_b,
                return_attention_weights=return_attention
            )
            
            if return_attention:
                # If we asked for weights, unpack the two results
                attention_result, weights_b = attention_output
                if weights_b is not None:
                    batched_attention_weights = weights_b
            else:
                # Otherwise, the single result is the tensor
                attention_result = attention_output
            
            # Remove the batch dimension
            attended_square_embeddings = attention_result.squeeze(1)

        # The original plan included a fusion layer. A simpler starting point is
        # to just use the attended square embeddings directly. This is a common
        # architecture (e.g., "use the output of the transformer").
        # The heads are designed to take this embedding dimension.
        fused_embeddings = attended_square_embeddings
        
        # Add a batch dimension for the heads
        fused_embeddings_for_heads = fused_embeddings.unsqueeze(0)

        policy_logits_b = self.policy_head(fused_embeddings_for_heads)
        value_b = self.value_head(fused_embeddings_for_heads)

        # Remove batch dimension for the final output
        policy_logits = policy_logits_b.squeeze(0)
        final_value = value_b.squeeze(0)

        if return_attention:
            return policy_logits, final_value, batched_attention_weights
        else:
            return policy_logits, final_value