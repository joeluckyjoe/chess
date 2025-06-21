import torch
import torch.nn as nn
from typing import Tuple, Optional

from .gnn_models import SquareGNN
from .gnn_models import PieceGNN
from .attention_module import CrossAttentionModule
from .policy_value_heads import PolicyHead, ValueHead # Corrected import

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

        # This projection layer fuses the two outputs of the symmetric attention
        # back into a single representation for the policy/value heads.
        # Input dimension is the sum of the two embedding dimensions.
        # Output dimension is the original square embedding dimension.
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
                piece_padding_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the full network using symmetric cross-attention.

        Args:
            square_features (torch.Tensor): Features for each of the 64 squares.
            square_edge_index (torch.Tensor): Edge index for the square GNN.
            piece_features (torch.Tensor): Features for pieces currently on the board.
            piece_edge_index (torch.Tensor): Edge index for the piece GNN.
            piece_to_square_map (torch.Tensor): Maps piece indices to their square indices (0-63).
            piece_padding_mask (torch.Tensor, optional): Mask for padded pieces in batched scenarios.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - policy_logits (torch.Tensor): Raw output from the policy head.
            - final_value (torch.Tensor): Scalar evaluation from the value head.
        """
        square_embeddings = self.square_gnn(square_features, square_edge_index)
        piece_embeddings = self.piece_gnn(piece_features, piece_edge_index)

        final_square_representation = square_embeddings

        # Only apply attention if there are pieces on the board.
        if piece_embeddings.numel() > 0:
            # Manually add a batch dimension of 1 for the attention module.
            # This is part of the established pattern for non-batched inference.
            sq_embed_b = square_embeddings.unsqueeze(1)
            pc_embed_b = piece_embeddings.unsqueeze(1)

            # The symmetric attention module returns two tensors.
            attended_piece_embeddings_b, attended_square_embeddings_b = self.cross_attention(
                square_embeddings=sq_embed_b,
                piece_embeddings=pc_embed_b,
                piece_padding_mask=piece_padding_mask
            )

            # Squeeze the batch dimension after attention.
            attended_piece_embeddings = attended_piece_embeddings_b.squeeze(1)
            attended_square_embeddings = attended_square_embeddings_b.squeeze(1)

            # --- Fusion of the two attention outputs ---

            # 1. Create the piece-centric board representation by placing attended
            #    piece embeddings onto the squares they occupy.
            piece_centric_board_representation = torch.zeros_like(square_embeddings)
            # This operation requires attended_piece_embeddings to have the same feature dimension
            # as square_embeddings. We need a projection if they differ.
            # Assuming pc_embed_dim == sq_embed_dim for this to work directly.
            # If not, a projection would be needed on attended_piece_embeddings.
            if attended_piece_embeddings.shape[1] != piece_centric_board_representation.shape[1]:
                 raise ValueError("Feature dimensions of piece and square embeddings must match for index_add_.")

            piece_centric_board_representation.index_add_(0, piece_to_square_map, attended_piece_embeddings)

            # 2. Concatenate the piece-centric board representation with the
            #    square-centric representation.
            fused_representation = torch.cat(
                [piece_centric_board_representation, attended_square_embeddings],
                dim=1
            )

            # 3. Apply the final projection layer to get the final board state.
            final_square_representation = self.relu(self.embedding_layer(fused_representation))

        # Add a batch dimension for the policy and value heads.
        final_representation_for_heads = final_square_representation.unsqueeze(0)

        policy_logits_b = self.policy_head(final_representation_for_heads)
        value_b = self.value_head(final_representation_for_heads)

        # Remove batch dimension for the final output.
        policy_logits = policy_logits_b.squeeze(0)
        final_value = value_b.squeeze(0)

        return policy_logits, final_value