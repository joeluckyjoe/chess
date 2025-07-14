#
# File: gnn_agent/neural_network/chess_network.py (Updated for GNN+CNN Hybrid)
#
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel  # Import the new CNN model
from .gnn_models import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main PyTorch module for the chess agent.
    This version implements the GNN+CNN hybrid architecture. It processes board
    state through two parallel paths—a GNN for relational reasoning and a CNN
    for spatial pattern recognition—and fuses their outputs.
    """
    def __init__(self, gnn_embed_dim: int = 256, cnn_embed_dim: int = 256, gnn_hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()

        # --- GNN Path ---
        metadata = (
            ['square', 'piece'],
            [
                ('square', 'adjacent_to', 'square'),
                ('piece', 'occupies', 'square'),
                ('piece', 'attacks', 'piece'),
                ('piece', 'defends', 'piece'),
                ('square', 'rev_occupied_by', 'piece'),
            ]
        )
        self.unified_gnn = UnifiedGNN(
            hidden_dim=gnn_hidden_dim,
            embed_dim=gnn_embed_dim,
            num_heads=num_heads,
            metadata=metadata
        )

        # --- CNN Path ---
        # Assuming 14 input channels for the 2D board representation
        self.cnn_model = CNNModel(in_channels=14, embedding_dim=cnn_embed_dim)

        # --- Fusion and Head Architecture ---
        fused_dim = gnn_embed_dim + cnn_embed_dim
        trunk_dim = fused_dim // 2
        
        self.policy_trunk = nn.Sequential(
            nn.Linear(fused_dim, trunk_dim),
            nn.GELU(),
        )
        self.value_trunk = nn.Sequential(
            nn.Linear(fused_dim, trunk_dim),
            nn.GELU(),
        )

        self.policy_head = PolicyHead(trunk_dim)
        self.value_head = ValueHead(trunk_dim)

    def forward(self, gnn_data: Batch, cnn_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass for the GNN+CNN hybrid model.

        Args:
            gnn_data (Batch): A PyG Batch object containing graph data for the batch.
            cnn_data (torch.Tensor): A tensor for the CNN of shape 
                                     (batch_size, channels, 8, 8).

        Returns:
            A tuple containing policy logits and the value estimate.
        """
        # 1. Process data through the GNN path
        # piece_embeds shape: [total_num_pieces_in_batch, gnn_embed_dim]
        piece_embeds = self.unified_gnn(gnn_data)
        
        # Aggregate GNN embeddings to get one vector per graph
        # gnn_embedding shape: [batch_size, gnn_embed_dim]
        gnn_embedding = global_mean_pool(piece_embeds, gnn_data['piece'].batch)

        # 2. Process data through the CNN path
        # cnn_embedding shape: [batch_size, cnn_embed_dim]
        cnn_embedding = self.cnn_model(cnn_data)

        # 3. Fuse the embeddings from both paths
        # fused_embedding shape: [batch_size, gnn_embed_dim + cnn_embed_dim]
        fused_embedding = torch.cat([gnn_embedding, cnn_embedding], dim=1)

        # 4. Specialize the fused embedding for each task
        policy_embed = self.policy_trunk(fused_embedding)
        value_embed = self.value_trunk(fused_embedding)

        # 5. Pass specialized embeddings to the final heads
        policy_logits = self.policy_head(policy_embed)
        value_estimate = self.value_head(value_embed)

        return policy_logits, value_estimate