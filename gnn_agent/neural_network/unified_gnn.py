#
# File: gnn_agent/neural_network/unified_gnn.py (Corrected for Pylance error)
#
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, global_max_pool
from torch_geometric.data import HeteroData

from ..gamestate_converters.gnn_data_converter import SQUARE_FEATURE_DIM, PIECE_FEATURE_DIM

class UnifiedGNN(nn.Module):
    """
    A Unified Heterogeneous Graph Neural Network for processing a chess board state.
    This version is refactored to use HeteroConv, which is more robust to fx tracing
    errors than the previous to_hetero wrapper.
    """
    def __init__(self, hidden_dim: int, embed_dim: int, num_heads: int, metadata: tuple):
        super().__init__()
        
        # --- FIX: Store metadata as a class attribute ---
        self.metadata = metadata
        
        self.lin_piece = nn.Linear(PIECE_FEATURE_DIM, hidden_dim)
        self.lin_square = nn.Linear(SQUARE_FEATURE_DIM, hidden_dim)
        
        self.conv1 = HeteroConv({
            edge_type: GATv2Conv((-1, -1), hidden_dim, heads=num_heads, concat=True, add_self_loops=False)
            for edge_type in self.metadata[1]
        }, aggr='sum')

        self.conv2 = HeteroConv({
            edge_type: GATv2Conv((-1, -1), embed_dim, heads=num_heads, concat=False, add_self_loops=False)
            for edge_type in self.metadata[1]
        }, aggr='sum')

        self.activation = nn.GELU()

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Forward pass for the UnifiedGNN.
        """
        x_dict = {
            "piece": self.activation(self.lin_piece(data["piece"].x)),
            "square": self.activation(self.lin_square(data["square"].x))
        }

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, data.edge_index_dict)

        piece_embeds = x_dict["piece"]
        piece_batch = data["piece"].batch
        if piece_embeds.numel() == 0:
            batch_size = data["square"].batch.max().item() + 1
            # --- FIX: Reference the stored self.metadata ---
            final_embed_dim = self.conv2.convs[self.metadata[1][0]].out_channels
            return torch.zeros(batch_size, final_embed_dim, device=piece_embeds.device)

        global_graph_embed = global_max_pool(piece_embeds, piece_batch.to(piece_embeds.device))

        return global_graph_embed