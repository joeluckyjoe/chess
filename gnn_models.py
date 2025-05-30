# gnn_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv

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
# Define these constants based on your GNNDataConverter output and desired embedding sizes.
# These are example values.
NUM_PIECE_FEATURES = 8  # e.g., 6 for piece type (one-hot) + 2 for color (one-hot)
PIECE_HIDDEN_CHANNELS = 16 # Example intermediate dimension
PIECE_EMBEDDING_DIM = 32   # Should ideally match SQUARE_EMBEDDING_DIM

class PieceGNN(torch.nn.Module):
    def __init__(self, in_channels=NUM_PIECE_FEATURES, hidden_channels=PIECE_HIDDEN_CHANNELS, out_channels=PIECE_EMBEDDING_DIM):
        super(PieceGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x_piece, edge_index_piece):
        """
        Forward pass for the PieceGNN.

        Args:
            x_piece (torch.Tensor): Piece features of shape [num_pieces, num_piece_features].
                                    num_pieces can vary.
            edge_index_piece (torch.Tensor): Piece graph connectivity in COO format
                                             of shape [2, num_piece_edges].

        Returns:
            torch.Tensor: Piece embeddings of shape [num_pieces, piece_embedding_dim].
        """
        # Handle cases with no pieces or no edges if necessary,
        # though PyG layers often handle this gracefully.
        if x_piece is None or x_piece.size(0) == 0:
            # Return an empty tensor with the correct embedding dimension
            return torch.empty((0, self.conv2.out_channels), device=x_piece.device if x_piece is not None else 'cpu')

        x = self.conv1(x_piece, edge_index_piece)
        x = F.relu(x)
        x = self.conv2(x, edge_index_piece)
        # No final activation, allowing embeddings to be in any range,
        # or you could add one e.g. F.relu(x) or torch.tanh(x) if desired.
        return x
