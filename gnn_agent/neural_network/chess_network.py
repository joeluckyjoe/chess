#
# File: gnn_agent/neural_network/chess_network.py (Updated with separate trunks)
#
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .gnn_models import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main PyTorch module for the chess agent.
    This version includes separate trunks for the policy and value heads to
    decouple their learning objectives and improve convergence.
    """
    def __init__(self, embed_dim: int = 256, gnn_hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()

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
            embed_dim=embed_dim,
            num_heads=num_heads,
            metadata=metadata
        )
        
        # --- MODIFICATION: Add separate trunks for each head ---
        trunk_dim = embed_dim // 2 # Intermediate dimension for the trunks
        
        self.policy_trunk = nn.Sequential(
            nn.Linear(embed_dim, trunk_dim),
            nn.GELU(),
        )
        self.value_trunk = nn.Sequential(
            nn.Linear(embed_dim, trunk_dim),
            nn.GELU(),
        )
        # --- END MODIFICATION ---

        # The policy and value heads now take the trunk's output as input
        self.policy_head = PolicyHead(trunk_dim)
        self.value_head = ValueHead(trunk_dim)

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass now pipes the shared embedding through separate trunks.
        """
        # 1. Get the shared embedding from the main GNN
        shared_embed = self.unified_gnn(data)

        # 2. Specialize the embedding for each task using the trunks
        policy_embed = self.policy_trunk(shared_embed)
        value_embed = self.value_trunk(shared_embed)

        # 3. Pass specialized embeddings to the final heads
        policy_logits = self.policy_head(policy_embed)
        value_estimate = self.value_head(value_embed)

        return policy_logits, value_estimate