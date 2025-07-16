#
# File: gnn_agent/neural_network/unified_gnn.py (Corrected for Hybrid Fusion)
#
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData

from ..gamestate_converters.gnn_data_converter import SQUARE_FEATURE_DIM, PIECE_FEATURE_DIM

class UnifiedGNN(nn.Module):
    """
    A Unified Heterogeneous Graph Neural Network for processing a chess board state.
    
    MODIFICATION: This version now correctly returns the per-SQUARE embeddings
    to ensure its output is compatible with the per-square CNN feature map.
    """
    def __init__(self, hidden_dim: int, embed_dim: int, num_heads: int, metadata: tuple):
        super().__init__()
        
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

        # --- FINAL CORRECTION: Return the per-square embeddings ---
        # This ensures the output tensor always corresponds to the 64 squares.
        return x_dict["square"]
