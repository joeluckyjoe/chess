#
# File: gnn_agent/neural_network/unified_gnn.py (Corrected)
#
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, global_max_pool
from torch_geometric.data import HeteroData

from ..gamestate_converters.gnn_data_converter import SQUARE_FEATURE_DIM, PIECE_FEATURE_DIM

class UnifiedGNN(nn.Module):
    """
    A Unified Heterogeneous Graph Neural Network for processing a chess board state.
    This module replaces the previous dual-GNN + cross-attention architecture.
    It uses HGTConv layers to process a single graph with 'piece' and 'square' nodes.
    """
    def __init__(self, hidden_dim: int, embed_dim: int, num_heads: int, metadata: tuple):
        super().__init__()
        
        self.lin_piece = nn.Linear(PIECE_FEATURE_DIM, hidden_dim)
        self.lin_square = nn.Linear(SQUARE_FEATURE_DIM, hidden_dim)

        self.conv1 = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=num_heads
        )

        self.conv2 = HGTConv(
            in_channels=hidden_dim,
            out_channels=embed_dim,
            metadata=metadata,
            heads=num_heads
        )

        # --- CORRECTED ACTIVATION ---
        # Replaced ReLU with GELU to align with Phase AI findings and prevent saturation.
        self.activation = nn.GELU()
        # --- END CORRECTION ---


    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Forward pass for the UnifiedGNN.
        """
        # 1. Apply initial linear projections and activation
        x_dict = {
            "piece": self.activation(self.lin_piece(data["piece"].x)),
            "square": self.activation(self.lin_square(data["square"].x))
        }

        # 2. First HGT convolution
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}

        # 3. Second HGT convolution
        x_dict = self.conv2(x_dict, data.edge_index_dict)

        # 4. Global Pooling
        piece_embeds = x_dict["piece"]
        piece_batch = data["piece"].batch
        if piece_embeds.numel() == 0:
            batch_size = data["square"].batch.max().item() + 1
            return torch.zeros(batch_size, piece_embeds.size(-1), device=piece_embeds.device)

        global_graph_embed = global_max_pool(piece_embeds, piece_batch.to(piece_embeds.device))

        return global_graph_embed