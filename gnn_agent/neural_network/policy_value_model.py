import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple

# Project-specific imports
from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel

class PolicyValueModel(nn.Module):
    """
    A GNN+CNN hybrid model for chess that outputs a policy and a value.
    This is the simplified two-headed architecture for Phase BR.
    """
    def __init__(self,
                 gnn_hidden_dim: int,
                 cnn_in_channels: int,
                 embed_dim: int,
                 policy_size: int,
                 gnn_num_heads: int,
                 gnn_metadata: tuple):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Core feature extractors
        self.gnn = UnifiedGNN(hidden_dim=gnn_hidden_dim, embed_dim=embed_dim, num_heads=gnn_num_heads, metadata=gnn_metadata)
        self.cnn = CNNModel(in_channels=cnn_in_channels, embedding_dim=embed_dim)
        
        # Fusion and projection layer for combined features
        fused_dim = 2 * embed_dim
        self.embedding_projection = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # Output Heads
        self.policy_head = nn.Linear(embed_dim, policy_size)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, gnn_batch: Batch, cnn_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the network.
        
        Args:
            gnn_batch: A PyTorch Geometric Batch object for the GNN.
            cnn_tensor: A tensor of shape (batch_size, channels, 8, 8) for the CNN.

        Returns:
            A tuple containing policy logits and the state value.
        """
        batch_size = cnn_tensor.size(0)
        
        # Get embeddings from GNN and CNN
        gnn_out = self.gnn(gnn_batch)
        cnn_out = self.cnn(cnn_tensor)
        
        # Reshape and pool GNN output
        gnn_out_reshaped = gnn_out.view(batch_size, 64, self.embed_dim)
        gnn_out_pooled = gnn_out_reshaped.mean(dim=1)
        
        # Reshape and pool CNN output
        cnn_out_flat = cnn_out.view(batch_size, self.embed_dim, -1)
        cnn_out_pooled = cnn_out_flat.mean(dim=2)
        
        # Fuse the embeddings from both models
        fused = torch.cat([gnn_out_pooled, cnn_out_pooled], dim=-1)
        
        # Project the fused embedding into a shared representation
        final_embedding = self.embedding_projection(fused)
        
        # Calculate policy and value
        policy_logits = self.policy_head(final_embedding)
        value = torch.tanh(self.value_head(final_embedding))
        
        return policy_logits, value