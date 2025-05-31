import torch
import torch.nn as nn
import torch.nn.functional as F
# numpy is not directly used in this file, but good to keep if future versions might
# import numpy as np 

class GCNLayer(nn.Module):
    """
    A Graph Convolutional Network layer with symmetric normalization.
    """
    def __init__(self, input_features: int, output_features: int, use_bias: bool = True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(input_features, output_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GCN layer.

        Args:
            node_features (torch.Tensor): Node features of shape (num_nodes, input_features).
            adj_matrix (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).
                                       Should be the raw adjacency matrix.
        Returns:
            torch.Tensor: Output node features of shape (num_nodes, output_features).
        """
        num_nodes = adj_matrix.size(0)
        
        # 1. Add self-loops: A_hat = A + I
        A_hat = adj_matrix + torch.eye(num_nodes, device=adj_matrix.device)
        
        # 2. Calculate degree matrix D_hat
        # D_hat_ii = sum_j A_hat_ij
        D_hat_diag = torch.sum(A_hat, dim=1)
        
        # 3. Compute D_hat^(-1/2)
        # Avoid division by zero for isolated nodes (though unlikely in our 8-way/fully_connected)
        D_hat_inv_sqrt_diag = torch.pow(D_hat_diag, -0.5)
        D_hat_inv_sqrt_diag[torch.isinf(D_hat_inv_sqrt_diag)] = 0. # Replace inf with 0
        
        # Create sparse diagonal matrix for D_hat_inv_sqrt if performance becomes an issue
        # For now, using a dense diagonal matrix multiplication is fine for 64 nodes.
        D_hat_inv_sqrt = torch.diag(D_hat_inv_sqrt_diag)
        
        # 4. Compute the normalized adjacency matrix: D_hat^(-1/2) * A_hat * D_hat^(-1/2)
        # adj_normalized = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt # @ is matmul
        # More efficiently for sparse, but for dense:
        adj_normalized = torch.matmul(D_hat_inv_sqrt, torch.matmul(A_hat, D_hat_inv_sqrt))
        
        # 5. Perform the GCN operation
        support = torch.matmul(node_features, self.weight) # H * W
        output = torch.matmul(adj_normalized, support)     # D_hat_inv_sqrt * A_hat * D_hat_inv_sqrt * H * W
        
        if self.bias is not None:
            output = output + self.bias
        return output

class GNN_sq_BaseModel(nn.Module):
    """
    A simple GNN model for encoding square features.
    """
    def __init__(self, input_feature_dim: int, gnn_embedding_dim: int = 32):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.gnn_embedding_dim = gnn_embedding_dim

        self.gcn1 = GCNLayer(input_feature_dim, gnn_embedding_dim)
        # Potentially add more layers or different types of layers later

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features (torch.Tensor): Shape (num_squares, input_feature_dim)
            adj_matrix (torch.Tensor): Shape (num_squares, num_squares)

        Returns:
            torch.Tensor: Square embeddings, shape (num_squares, gnn_embedding_dim)
        """
        # Ensure adj_matrix is float for matmul if it's not already
        adj_matrix = adj_matrix.float() 
        
        x = self.gcn1(node_features, adj_matrix)
        x = F.relu(x)
        # In a more complex model, you might have more GCN layers, pooling, etc.
        return x