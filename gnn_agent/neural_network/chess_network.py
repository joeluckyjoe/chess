#
# File: gnn_agent/neural_network/chess_network.py (Corrected for new UnifiedGNN)
#
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .gnn_models import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main PyTorch module for the chess agent.
    This version correctly integrates the GNN and Transformer, with proper
    batch handling for the MCTS.
    """
    def __init__(self, embed_dim: int = 256, gnn_hidden_dim: int = 128, num_heads: int = 4, transformer_layers: int = 2):
        super().__init__()

        # This metadata must match the metadata used in the UnifiedGNN
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
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=transformer_layers
        )

        trunk_dim = embed_dim // 2
        
        self.policy_trunk = nn.Sequential(
            nn.Linear(embed_dim, trunk_dim),
            nn.GELU(),
        )
        self.value_trunk = nn.Sequential(
            nn.Linear(embed_dim, trunk_dim),
            nn.GELU(),
        )

        self.policy_head = PolicyHead(trunk_dim)
        self.value_head = ValueHead(trunk_dim)

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass with correct GNN -> Transformer data flow.
        """
        # 1. Get the sequence of per-piece embeddings from the GNN.
        # Shape: [total_num_pieces_in_batch, embed_dim]
        piece_embeds = self.unified_gnn(data)
        
        # 2. Convert the sparse PyG batch to a dense tensor for the Transformer.
        # Shape: [batch_size, max_pieces, embed_dim], Mask: [batch_size, max_pieces]
        dense_piece_embeds, mask = to_dense_batch(piece_embeds, batch=data['piece'].batch)
        
        # The transformer's padding mask expects True for positions to be *ignored*.
        padding_mask = ~mask

        # 3. Pass the dense batch through the Transformer, using the padding mask.
        transformer_output = self.transformer_encoder(dense_piece_embeds, src_key_padding_mask=padding_mask)

        # 4. Aggregate the Transformer's output via masked average pooling.
        # We replace padded values with 0 so they don't contribute to the sum.
        transformer_output = transformer_output.masked_fill(padding_mask.unsqueeze(-1), 0)
        num_pieces = mask.sum(dim=1, keepdim=True)
        # Avoid division by zero for edge cases (e.g., a board with no pieces)
        num_pieces = torch.clamp(num_pieces, min=1)
        
        aggregated_embed = transformer_output.sum(dim=1) / num_pieces

        # 5. Specialize the aggregated embedding for each task.
        policy_embed = self.policy_trunk(aggregated_embed)
        value_embed = self.value_trunk(aggregated_embed)

        # 6. Pass specialized embeddings to the final heads.
        policy_logits = self.policy_head(policy_embed)
        value_estimate = self.value_head(value_embed)

        return policy_logits, value_estimate