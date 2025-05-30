# gnn_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class SquareGNN(nn.Module):
    """
    A Graph Attention Network (GAT) for processing the 64 squares of a chessboard.

    This GNN takes the graph representation of the board's squares, where each of
    the 64 nodes has features representing the piece on it (if any), and
    produces a rich embedding for each square.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, heads: int = 4):
        """
        Initializes the SquareGNN model.

        Args:
            in_features (int): The number of input features for each node (square).
                               This corresponds to the length of the feature vector
                               from the GnnDataConverter.
            hidden_features (int): The number of features in the hidden layer.
            out_features (int): The number of output features for each node.
                               This is the dimension of the final square embedding.
            heads (int, optional): The number of multi-head attentions to use in
                                   the GATv2Conv layers. Defaults to 4.
        """
        super(SquareGNN, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.heads = heads

        # First Graph Attention Layer
        # Note: The output dimension of a multi-head GAT layer is heads * out_channels.
        self.conv1 = GATv2Conv(in_features, hidden_features, heads=heads, concat=True, dropout=0.6)

        # Second Graph Attention Layer
        # The input to this layer is the concatenated output of the previous layer's heads.
        self.conv2 = GATv2Conv(hidden_features * heads, out_features, heads=1, concat=True, dropout=0.6)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the SquareGNN.

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_features].
                              For the square graph, num_nodes is always 64.
            edge_index (torch.Tensor): Graph connectivity in COO format with shape
                                       [2, num_edges].

        Returns:
            torch.Tensor: The output node embeddings with shape [num_nodes, out_features].
        """
        # Apply dropout after the input layer (common practice)
        x = F.dropout(x, p=0.6, training=self.training)

        # Apply first GAT layer and GELU activation
        x = self.conv1(x, edge_index)
        x = F.gelu(x)

        # Apply dropout
        x = F.dropout(x, p=0.6, training=self.training)

        # Apply second GAT layer
        x = self.conv2(x, edge_index)

        return x