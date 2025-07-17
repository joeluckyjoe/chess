import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel

class HybridRNNModel(nn.Module):
    """
    CORRECTED: This version processes a BATCH of individual board states and a
    corresponding BATCH of hidden states, making it compatible with the MCTS batch
    evaluation loop.
    """
    def __init__(self, gnn_hidden_dim: int, cnn_in_channels: int, embed_dim: int, num_heads: int, gnn_metadata: tuple, rnn_hidden_dim: int, num_rnn_layers: int, policy_size: int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers

        self.gnn = UnifiedGNN(hidden_dim=gnn_hidden_dim, embed_dim=embed_dim, num_heads=num_heads, metadata=gnn_metadata)
        self.cnn = CNNModel(in_channels=cnn_in_channels, embedding_dim=embed_dim)

        fused_dim = 2 * embed_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim)
        )
        
        self.rnn = nn.GRU(
            input_size=fused_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True
        )

        self.policy_head = nn.Linear(rnn_hidden_dim, policy_size)
        self.value_head = nn.Linear(rnn_hidden_dim, 1)

    def forward(self, gnn_batch: Batch, cnn_batch: torch.Tensor, hidden_state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a batch of board states, each with its corresponding hidden state.
        
        Args:
            gnn_batch (torch_geometric.data.Batch): A batch of GNN data.
            cnn_batch (torch.Tensor): A batch of CNN data of shape (N, C, 8, 8).
            hidden_state_batch (torch.Tensor): The batch of hidden states for the GRU, shape (num_layers, N, rnn_hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Policy logits, value, and the new hidden state.
        """
        # 1. Get per-square embeddings from GNN and CNN
        gnn_out = self.gnn(gnn_batch)
        cnn_out_map = self.cnn(cnn_batch)
        
        # 2. Reshape and Pool
        batch_size = cnn_batch.size(0)
        gnn_out_reshaped = gnn_out.view(batch_size, 64, self.embed_dim)
        gnn_out_pooled = gnn_out_reshaped.mean(dim=1)

        cnn_out_flat = cnn_out_map.view(batch_size, self.embed_dim, -1)
        cnn_out_pooled = cnn_out_flat.mean(dim=2)

        # 3. Fuse, unsqueeze for RNN, and pass through MLP
        fused = torch.cat([gnn_out_pooled, cnn_out_pooled], dim=-1)
        fused_for_rnn = fused.unsqueeze(1)
        mlp_out = self.fusion_mlp(fused_for_rnn)

        # 4. Pass through RNN
        rnn_output, new_hidden_state = self.rnn(mlp_out, hidden_state_batch)
        
        # 5. Get policy and value from RNN output
        last_step_output = rnn_output.squeeze(1)
        policy_logits = self.policy_head(last_step_output)
        value = torch.tanh(self.value_head(last_step_output))

        return policy_logits, value, new_hidden_state