#
# File: gnn_models.py (Updated for Phase AK)
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
from typing import Optional

class SquareGNN(nn.Module):
    """
    A Graph Attention Network (GAT) for processing the 64 squares of a chessboard.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, heads: int = 4):
        super(SquareGNN, self).__init__()
        self.conv1 = GATv2Conv(in_features, hidden_features, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_features * heads, out_features, heads=1, concat=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        return x

class PieceGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PieceGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # --- PHASE AK MODIFICATION: Add layer to create 3-layer GNN ---
        self.conv2 = GCNConv(hidden_channels, hidden_channels) # Intermediate layer
        self.conv3 = GCNConv(hidden_channels, out_channels) # Output layer
        # --- END MODIFICATION ---

    def forward(self, x_piece: torch.Tensor, edge_index_piece: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the PieceGNN.
        """
        # --- PHASE AK MODIFICATION: Update output channel reference for empty tensor case ---
        if x_piece is None or x_piece.size(0) == 0:
            return torch.empty((0, self.conv3.out_channels), device=x_piece.device if x_piece is not None else 'cpu')
        # --- END MODIFICATION ---

        if edge_index_piece.size(1) == 0:
            num_nodes = x_piece.size(0)
            # --- PHASE AK MODIFICATION: Update output channel reference for no-edge case ---
            out_dim = self.conv3.out_channels
            # --- END MODIFICATION ---
            return torch.zeros((num_nodes, out_dim), device=x_piece.device)

        x = self.conv1(x_piece, edge_index_piece)
        x = F.gelu(x)
        
        # --- PHASE AK MODIFICATION: Add forward pass for the new layer ---
        x = self.conv2(x, edge_index_piece)
        x = F.gelu(x)
        x = self.conv3(x, edge_index_piece)
        # --- END MODIFICATION ---
        return x