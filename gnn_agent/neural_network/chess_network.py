#
# File: gnn_agent/neural_network/chess_network.py (Updated with Transformer)
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
    
    Phase BE Modification: This version adds a TransformerEncoder layer after
    the UnifiedGNN to provide a mechanism for global, all-to-all reasoning,
    aiming to break the "plateau-intervene-improve" cycle.
    """
    def __init__(self, embed_dim: int = 256, gnn_hidden_dim: int = 128, num_heads: int = 4, transformer_layers: int = 2):
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
        
        # --- MODIFICATION: Add TransformerEncoder Layer ---
        # The hypothesis is that the GNN's local message-passing is insufficient.
        # A Transformer can weigh the importance of all piece embeddings against
        # each other simultaneously, providing global context.
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, # Standard practice
            dropout=0.1,
            activation='gelu', # Consistent with our other activations
            batch_first=True # Important for our data shape
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=transformer_layers
        )
        # --- END MODIFICATION ---

        # Separate trunks for each head to decouple learning objectives.
        trunk_dim = embed_dim // 2 # Intermediate dimension for the trunks
        
        self.policy_trunk = nn.Sequential(
            nn.Linear(embed_dim, trunk_dim),
            nn.GELU(),
        )
        self.value_trunk = nn.Sequential(
            nn.Linear(embed_dim, trunk_dim),
            nn.GELU(),
        )

        # The policy and value heads now take the trunk's output as input
        self.policy_head = PolicyHead(trunk_dim)
        self.value_head = ValueHead(trunk_dim)

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass now pipes the shared embedding through the Transformer.
        """
        # 1. Get the shared embedding from the main GNN
        shared_embed = self.unified_gnn(data)

        # --- MODIFICATION: Apply Transformer for global reasoning ---
        # The GNN output is (batch_size, num_pieces, embed_dim).
        # We need to add a batch dimension if it's not there for the transformer.
        if shared_embed.dim() == 2:
            shared_embed = shared_embed.unsqueeze(0) # (1, num_pieces, embed_dim)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(shared_embed)
        
        # We need to aggregate the transformer's output back to a single vector
        # per graph. A simple mean over the sequence dimension is a robust choice.
        # (batch_size, num_pieces, embed_dim) -> (batch_size, embed_dim)
        aggregated_embed = transformer_output.mean(dim=1)
        # --- END MODIFICATION ---

        # 2. Specialize the aggregated embedding for each task using the trunks
        policy_embed = self.policy_trunk(aggregated_embed)
        value_embed = self.value_trunk(aggregated_embed)

        # 3. Pass specialized embeddings to the final heads
        policy_logits = self.policy_head(policy_embed)
        value_estimate = self.value_head(value_embed)

        return policy_logits, value_estimate