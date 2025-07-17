import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel

class HybridRNNModel(nn.Module):
    """
    CORRECTED: This version re-integrates the material_head to provide an auxiliary
    loss signal, combining temporal context with concrete board evaluation.
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
        # --- RE-INTEGRATED: Material Balance Head ---
        self.material_head = nn.Linear(rnn_hidden_dim, 1)

    def forward(self, gnn_batch: Batch, cnn_batch: torch.Tensor, hidden_state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a batch of board states, returning policy, value, material balance, and new hidden state.
        """
        # (No change to this part of the forward pass)
        gnn_out = self.gnn(gnn_batch)
        cnn_out_map = self.cnn(cnn_batch)
        batch_size = cnn_batch.size(0)
        gnn_out_reshaped = gnn_out.view(batch_size, 64, self.embed_dim)
        gnn_out_pooled = gnn_out_reshaped.mean(dim=1)
        cnn_out_flat = cnn_out_map.view(batch_size, self.embed_dim, -1)
        cnn_out_pooled = cnn_out_flat.mean(dim=2)
        fused = torch.cat([gnn_out_pooled, cnn_out_pooled], dim=-1)
        fused_for_rnn = fused.unsqueeze(1)
        mlp_out = self.fusion_mlp(fused_for_rnn)
        rnn_output, new_hidden_state = self.rnn(mlp_out, hidden_state_batch)
        
        last_step_output = rnn_output.squeeze(1)
        
        # --- MODIFIED: Return predictions from all three heads ---
        policy_logits = self.policy_head(last_step_output)
        value = torch.tanh(self.value_head(last_step_output))
        material_balance = self.material_head(last_step_output)

        return policy_logits, value, material_balance, new_hidden_state