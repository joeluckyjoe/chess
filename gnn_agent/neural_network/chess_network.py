#
# File: gnn_agent/neural_network/chess_network.py (Corrected for Phase BI)
#
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel
from .policy_value_heads import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main PyTorch module for the chess agent. (Corrected for Phase BI)
    
    This version implements the GNN+CNN hybrid architecture. It processes board
    state through two parallel paths—a GNN for relational reasoning and a CNN
    for spatial pattern recognition—and fuses their outputs.
    
    The network now produces three outputs: policy logits, a game outcome value,
    and a material balance value.
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
        self.cnn_model = CNNModel(in_channels=14, embedding_dim=cnn_embed_dim)

        # --- Fusion and Head Architecture ---
        fused_dim = gnn_embed_dim + cnn_embed_dim
        
        # The policy and value trunks operate on the fused per-square embeddings
        self.policy_trunk = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
        )
        self.value_trunk = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
        )

        self.policy_head = PolicyHead(fused_dim // 2)
        self.value_head = ValueHead(fused_dim // 2)

    def forward(self, gnn_data: Batch, cnn_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass for the GNN+CNN hybrid model. (Corrected for Phase BI)

        Args:
            gnn_data (Batch): A PyG Batch object containing graph data for the batch.
            cnn_data (torch.Tensor): A tensor for the CNN of shape 
                                     (batch_size, channels, 8, 8).

        Returns:
            A tuple containing:
                - policy_logits (torch.Tensor)
                - value_estimate (torch.Tensor)
                - material_balance (torch.Tensor)
        """
        # --- ARCHITECTURE FIX ---
        # The previous version aggregated embeddings too early. The policy/value heads
        # expect per-square embeddings to work with their convolutional layers.
        # The corrected flow processes embeddings on a per-square basis until the final head.

        # 1. Process data through the GNN path to get per-square embeddings.
        #    The UnifiedGNN is assumed to return a dictionary of node-type embeddings.
        #    square_embeds_gnn shape: [total_squares_in_batch, gnn_embed_dim]
        gnn_output_dict = self.unified_gnn(gnn_data)
        square_embeds_gnn = gnn_output_dict['square']

        # 2. Process data through the CNN path.
        #    The CNN model should output per-square features, not a single vector.
        #    cnn_feature_map shape: [batch_size, cnn_embed_dim, 8, 8]
        cnn_feature_map = self.cnn_model(cnn_data)
        
        # Reshape CNN output to match the GNN's per-square format.
        # [B, D, 8, 8] -> [B, 8, 8, D] -> [B*64, D]
        batch_size, cnn_embed_dim, _, _ = cnn_feature_map.shape
        square_embeds_cnn = cnn_feature_map.permute(0, 2, 3, 1).reshape(batch_size * 64, cnn_embed_dim)

        # 3. Fuse the per-square embeddings from both paths.
        # fused_square_embeddings shape: [total_squares_in_batch, gnn_embed_dim + cnn_embed_dim]
        fused_square_embeddings = torch.cat([square_embeds_gnn, square_embeds_cnn], dim=1)

        # 4. Specialize the fused per-square embeddings for each task.
        policy_embeds = self.policy_trunk(fused_square_embeddings)
        value_embeds = self.value_trunk(fused_square_embeddings)

        # 5. Pass specialized per-square embeddings to the final heads.
        #    Crucially, we must also pass the batch tensor so the heads know how
        #    to group the squares for the final aggregation.
        square_batch_idx = gnn_data['square'].batch
        policy_logits = self.policy_head(policy_embeds, batch=square_batch_idx)
        value_estimate, material_balance = self.value_head(value_embeds, batch=square_batch_idx)

        return policy_logits, value_estimate, material_balance
        # --- END ARCHITECTURE FIX ---
