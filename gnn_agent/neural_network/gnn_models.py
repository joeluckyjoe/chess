#
# File: gnn_models.py (Corrected)
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
        self.conv1 = GATv2Conv(in_features, hidden_features, heads=heads, concat=True, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_features * heads, out_features, heads=1, concat=True, dropout=0.6)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.dropout(x, p=0.6, training=self.training)
        # The 'batch' tensor is implicitly used by the PyG convolution layers
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        # The 'batch' tensor is implicitly used by the PyG convolution layers
        x = self.conv2(x, edge_index)
        return x

class PieceGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PieceGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    # PHASE AB CORRECTION: Changed 'batch_piece' to 'batch' to match keyword argument
    def forward(self, x_piece: torch.Tensor, edge_index_piece: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the PieceGNN.
        """
        # Handle cases with no piece nodes
        if x_piece is None or x_piece.size(0) == 0:
            return torch.empty((0, self.conv2.out_channels), device=x_piece.device if x_piece is not None else 'cpu')

        # *** NEW SAFEGUARD ***
        # Handle case with nodes but no edges
        if edge_index_piece.size(1) == 0:
            # No edges means no information can propagate via graph convolutions.
            # Return a tensor of zeros, signifying no relational information.
            # The shape must match the expected output dimension.
            num_nodes = x_piece.size(0)
            out_dim = self.conv2.out_channels
            return torch.zeros((num_nodes, out_dim), device=x_piece.device)

        # The 'batch' tensor is implicitly used by the PyG convolution layers
        x = self.conv1(x_piece, edge_index_piece)
        x = F.relu(x)
        # The 'batch' tensor is implicitly used by the PyG convolution layers
        x = self.conv2(x, edge_index_piece)
        return x