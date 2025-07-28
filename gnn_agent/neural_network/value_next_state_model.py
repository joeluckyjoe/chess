# /home/giuseppe/chess/gnn_agent/neural_network/value_next_state_model.py
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple, Union

# Assuming these are in the same directory or the path is configured
from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel

class ValueNextStateModel(nn.Module):
    # ... (init method remains unchanged) ...
    def __init__(self,
                 gnn_hidden_dim: int,
                 cnn_in_channels: int,
                 embed_dim: int,
                 policy_size: int,
                 gnn_num_heads: int,
                 gnn_metadata: tuple):
        super().__init__()
        self.embed_dim = embed_dim
        self.gnn = UnifiedGNN(hidden_dim=gnn_hidden_dim, embed_dim=embed_dim, num_heads=gnn_num_heads, metadata=gnn_metadata)
        self.cnn = CNNModel(in_channels=cnn_in_channels, embedding_dim=embed_dim)
        fused_dim = 2 * embed_dim
        self.embedding_projection = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.policy_head = nn.Linear(embed_dim, policy_size)
        self.value_head = nn.Linear(embed_dim, 1)
        self.next_state_value_head = nn.Linear(embed_dim, 1)

    def forward(self, gnn_batch: Batch, cnn_tensor: torch.Tensor, return_embeddings: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Processes a batch of board states. Can optionally return GNN node embeddings.
        """
        batch_size = cnn_tensor.size(0)

        gnn_out = self.gnn(gnn_batch)
        cnn_out = self.cnn(cnn_tensor)

        gnn_out_reshaped = gnn_out.view(batch_size, 64, self.embed_dim)
        gnn_out_pooled = gnn_out_reshaped.mean(dim=1)

        cnn_out_flat = cnn_out.view(batch_size, self.embed_dim, -1)
        cnn_out_pooled = cnn_out_flat.mean(dim=2)

        fused = torch.cat([gnn_out_pooled, cnn_out_pooled], dim=-1)
        final_embedding = self.embedding_projection(fused)

        policy_logits = self.policy_head(final_embedding)
        value = torch.tanh(self.value_head(final_embedding))
        next_state_value = torch.tanh(self.next_state_value_head(final_embedding))

        if return_embeddings:
            return policy_logits, value, next_state_value, gnn_out
        else:
            return policy_logits, value, next_state_value