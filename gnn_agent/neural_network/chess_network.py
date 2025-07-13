#
# File: gnn_agent/neural_network/chess_network.py (Updated with Transformer & Batch Fix)
#
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import to_dense_batch
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .gnn_models import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main PyTorch module for the chess agent.
    
    Phase BE Modification: This version adds a TransformerEncoder layer after
    the UnifiedGNN to provide a mechanism for global, all-to-all reasoning,
    aiming to break the "plateau-intervene-improve" cycle.
    
    BUG FIX: Corrected the forward pass to handle PyTorch Geometric's
    batching mechanism, allowing it to work with the MCTS batch size.
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
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2, # Adjusted for efficiency
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
        The forward pass now pipes the shared embedding through the Transformer
        with corrected batch handling.
        """
        # 1. Get shared embeddings from the GNN.
        # This returns embeddings for all nodes in the batch concatenated together.
        gnn_output = self.unified_gnn(data)
        piece_embeds = gnn_output['piece'] # Shape: [total_num_pieces_in_batch, embed_dim]
        
        # --- BATCHING FIX ---
        # 2. Convert sparse PyG batch to a dense tensor for the Transformer.
        #    `to_dense_batch` creates a tensor of shape [batch_size, max_nodes, features]
        #    and a boolean mask to identify valid (non-padded) nodes.
        dense_piece_embeds, mask = to_dense_batch(piece_embeds, batch=data['piece'].batch)
        
        # The mask from to_dense_batch is True for valid nodes. The transformer's
        # padding mask expects True for positions to be *ignored*. So we invert it.
        padding_mask = ~mask

        # 3. Pass the dense batch through the Transformer, using the padding mask.
        transformer_output = self.transformer_encoder(dense_piece_embeds, src_key_padding_mask=padding_mask)

        # 4. Aggregate the Transformer's output. We perform a masked average pooling
        #    to get a single embedding per graph, ignoring the padded elements.
        #    We replace masked values with 0 so they don't contribute to the sum.
        transformer_output = transformer_output.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        # Summing over the sequence dimension (dim=1)
        # The mask needs to be summed to get the actual number of pieces per graph
        num_pieces = mask.sum(dim=1, keepdim=True)
        
        # Avoid division by zero for graphs with no pieces (should not happen in chess)
        num_pieces[num_pieces == 0] = 1
        
        aggregated_embed = transformer_output.sum(dim=1) / num_pieces
        # --- END OF FIX ---

        # 5. Specialize the aggregated embedding for each task using the trunks
        policy_embed = self.policy_trunk(aggregated_embed)
        value_embed = self.value_trunk(aggregated_embed)

        # 6. Pass specialized embeddings to the final heads
        policy_logits = self.policy_head(policy_embed)
        value_estimate = self.value_head(value_embed)

        return policy_logits, value_estimate