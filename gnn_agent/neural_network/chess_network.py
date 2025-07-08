# gnn_agent/neural_network/chess_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# --- CORRECTED IMPORT ---
from torch_geometric.nn import global_max_pool
# --- END CORRECTION ---

from .gnn_models import SquareGNN, PieceGNN, CrossAttentionModule, PolicyHead, ValueHead
from ..gamestate_converters.gnn_data_converter import SQUARE_FEATURE_DIM, PIECE_FEATURE_DIM


class ChessNetwork(nn.Module):
    """The main network orchestrating GNNs, attention, and heads."""
    def __init__(self, embed_dim: int = 256, gnn_hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.square_feature_dim = SQUARE_FEATURE_DIM
        self.piece_feature_dim = PIECE_FEATURE_DIM
        
        self.square_gnn = SquareGNN(SQUARE_FEATURE_DIM, gnn_hidden_dim, embed_dim, heads=num_heads)
        self.piece_gnn = PieceGNN(PIECE_FEATURE_DIM, embed_dim, embed_dim)
        self.fusion = CrossAttentionModule(embed_dim, num_heads)
        self.policy_head = PolicyHead(embed_dim)
        self.value_head = ValueHead(embed_dim)

    def forward(self, square_features, square_edge_index, square_batch,
                piece_features, piece_edge_index, piece_batch,
                piece_to_square_map, piece_padding_mask, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sq_embed = self.square_gnn(square_features, square_edge_index)
        pc_embed = self.piece_gnn(piece_features, piece_edge_index)

        if pc_embed.numel() > 0:
            sq_embed_fused, pc_embed_fused = self.fusion(sq_embed, pc_embed, piece_to_square_map)
            
            batch_size = square_batch.max().item() + 1
            global_graph_embed = global_max_pool(
                pc_embed_fused, 
                piece_batch.to(pc_embed_fused.device),
                size=batch_size
            )
        else:
            global_graph_embed = global_max_pool(sq_embed, square_batch)

        policy_logits = self.policy_head(global_graph_embed)
        value_estimate = self.value_head(global_graph_embed)

        return policy_logits, value_estimate.squeeze(-1)