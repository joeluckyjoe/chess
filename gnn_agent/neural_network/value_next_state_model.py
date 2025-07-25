# /home/giuseppe/chess/gnn_agent/neural_network/value_next_state_model.py
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel

class ValueNextStateModel(nn.Module):
    """
    Implements Phase BL: GNN+CNN with Policy, Value, and Next-State Value Heads.

    This model architecture is designed to cure the agent's strategic passivity
    by teaching it the immediate value consequences of its actions. It removes
    the temporal Transformer and focuses on a pure, single-state evaluation.
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

        # --- GNN and CNN Paths ---
        self.gnn = UnifiedGNN(hidden_dim=gnn_hidden_dim, embed_dim=embed_dim, num_heads=gnn_num_heads, metadata=gnn_metadata)
        self.cnn = CNNModel(in_channels=cnn_in_channels, embedding_dim=embed_dim)

        # --- Fusion Layer ---
        fused_dim = 2 * embed_dim
        self.embedding_projection = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # --- Output Heads ---
        self.policy_head = nn.Linear(embed_dim, policy_size)
        self.value_head = nn.Linear(embed_dim, 1)
        # New head to predict the value of the state resulting from a move
        self.next_state_value_head = nn.Linear(embed_dim, 1)

    def forward(self, gnn_batch: Batch, cnn_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a batch of individual board states for MCTS inference and training.

        Args:
            gnn_batch (torch_geometric.data.Batch): A batch of HeteroData objects.
            cnn_tensor (torch.Tensor): A tensor of shape (B, C, H, W).

        Returns:
            A tuple containing:
            - policy_logits (torch.Tensor): Raw scores for each possible move.
            - value (torch.Tensor): The estimated outcome of the current state (-1 to 1).
            - next_state_value (torch.Tensor): The estimated outcome of the next state (-1 to 1).
        """
        batch_size = cnn_tensor.size(0)

        # 1. Process GNN and CNN paths in parallel
        gnn_out = self.gnn(gnn_batch)
        cnn_out = self.cnn(cnn_tensor)

        # 2. Reshape and Pool to get one vector per board state
        # GNN output is (B * 64, embed_dim), needs reshaping
        gnn_out_reshaped = gnn_out.view(batch_size, 64, self.embed_dim)
        gnn_out_pooled = gnn_out_reshaped.mean(dim=1)

        # CNN output is (B, embed_dim, 8, 8), needs reshaping
        cnn_out_flat = cnn_out.view(batch_size, self.embed_dim, -1)
        cnn_out_pooled = cnn_out_flat.mean(dim=2)

        # 3. Fuse GNN and CNN embeddings
        fused = torch.cat([gnn_out_pooled, cnn_out_pooled], dim=-1)

        # 4. Project fused embedding into the model's main dimension
        final_embedding = self.embedding_projection(fused)

        # 5. Get predictions from all three heads
        policy_logits = self.policy_head(final_embedding)
        value = torch.tanh(self.value_head(final_embedding))
        next_state_value = torch.tanh(self.next_state_value_head(final_embedding))

        return policy_logits, value, next_state_value