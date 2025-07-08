# gnn_agent/neural_network/chess_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union

# --- Imports from gnn_models.py and other local files ---
from .gnn_models import SquareGNN, PieceGNN
from .attention_module import CrossAttentionModule
from .policy_value_heads import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main neural network that integrates the dual GNNs, symmetric cross-attention,
    and policy/value heads. This class is designed to process batches of game states
    for efficient training and MCTS evaluation.
    """
    def __init__(self,
                 square_gnn: SquareGNN,
                 piece_gnn: PieceGNN,
                 cross_attention: CrossAttentionModule,
                 policy_head: PolicyHead,
                 value_head: ValueHead):
        super(ChessNetwork, self).__init__()
        self.square_gnn = square_gnn
        self.piece_gnn = piece_gnn
        self.cross_attention = cross_attention
        self.policy_head = policy_head
        self.value_head = value_head
        try:
            embed_dim = value_head.fc1.in_features
            self.embedding_layer = nn.Linear(embed_dim * 2, embed_dim)
            self.activation = nn.GELU()
        except Exception:
            self.embedding_layer = nn.Identity()
            self.activation = nn.Identity()

    def forward(self,
                square_features: torch.Tensor,
                square_edge_index: torch.Tensor,
                square_batch: torch.Tensor,
                piece_features: torch.Tensor,
                piece_edge_index: torch.Tensor,
                piece_batch: torch.Tensor,
                piece_to_square_map: torch.Tensor,
                piece_padding_mask: torch.Tensor,
                return_attention: bool = False
               ) -> Union[Tuple[torch.Tensor, torch.Tensor],
                          Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        
        # The batch argument is NOT passed to the GNNs directly.
        square_embeddings = self.square_gnn(square_features, square_edge_index)
        piece_embeddings = self.piece_gnn(piece_features, piece_edge_index)

        final_square_representation = square_embeddings
        ps_attn_weights, sp_attn_weights = None, None
        
        if piece_embeddings.numel() > 0:
            batch_size = square_batch.max().item() + 1
            sq_embed_b = square_embeddings.view(batch_size, 64, -1)
            max_pieces_in_batch = piece_padding_mask.size(1)
            pc_embed_padded = torch.zeros(batch_size, max_pieces_in_batch, piece_embeddings.size(-1), device=piece_embeddings.device)
            _, counts = torch.unique_consecutive(piece_batch, return_counts=True)
            local_piece_indices = torch.cat([torch.arange(c) for c in counts]).to(piece_embeddings.device)
            flat_padded_indices = piece_batch * max_pieces_in_batch + local_piece_indices
            pc_embed_padded.view(-1, piece_embeddings.size(-1))[flat_padded_indices] = piece_embeddings
            pc_embed_b = pc_embed_padded
            attended_piece_embeddings_b, attended_square_embeddings_b, ps_attn_weights, sp_attn_weights = self.cross_attention(
                square_embeddings=sq_embed_b,
                piece_embeddings=pc_embed_b,
                piece_padding_mask=piece_padding_mask,
                return_attention=return_attention
            )
            attended_piece_embeddings = attended_piece_embeddings_b[~piece_padding_mask]
            attended_square_embeddings = attended_square_embeddings_b.view(-1, attended_square_embeddings_b.size(-1))
            piece_centric_board_representation = torch.zeros_like(square_embeddings)
            piece_centric_board_representation.index_add_(0, piece_to_square_map, attended_piece_embeddings)
            fused_representation = torch.cat(
                [piece_centric_board_representation, attended_square_embeddings],
                dim=1
            )
            final_square_representation = self.activation(self.embedding_layer(fused_representation))
        
        policy_logits = self.policy_head(final_square_representation, square_batch)
        value = self.value_head(final_square_representation, square_batch)

        if return_attention:
            return policy_logits, value, ps_attn_weights, sp_attn_weights
        else:
            return policy_logits, value