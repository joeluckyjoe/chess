import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel

class HybridTransformerModel(nn.Module):
    """
    Implements Phase BK: GNN+CNN+Transformer.

    MODIFIED: This version's forward pass is now polymorphic. It can handle:
    1. A sequence of board states for training (using the Transformer Encoder).
    2. A batch of individual board states for MCTS inference (bypassing the Transformer).
    """
    def __init__(self,
                 gnn_hidden_dim: int,
                 cnn_in_channels: int,
                 embed_dim: int,
                 policy_size: int,
                 gnn_num_heads: int,
                 transformer_nhead: int,
                 transformer_nlayers: int,
                 transformer_dim_feedforward: int,
                 gnn_metadata: tuple):
        super().__init__()

        self.embed_dim = embed_dim

        # --- GNN and CNN Paths ---
        self.gnn = UnifiedGNN(hidden_dim=gnn_hidden_dim, embed_dim=embed_dim, num_heads=gnn_num_heads, metadata=gnn_metadata)
        self.cnn = CNNModel(in_channels=cnn_in_channels, embedding_dim=embed_dim)

        # --- Fusion Layer ---
        fused_dim = 2 * embed_dim
        # MODIFIED: Renamed from fusion_mlp to reflect its new primary role
        self.embedding_projection = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # --- Transformer Encoder for Temporal Fusion ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)

        # --- Output Heads ---
        self.policy_head = nn.Linear(embed_dim, policy_size)
        self.value_head = nn.Linear(embed_dim, 1)
        self.material_head = nn.Linear(embed_dim, 1)

    def forward(self, gnn_batch: Batch, cnn_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes either a sequence of board states or a batch of individual states.
        - If cnn_tensor is 4D (B, C, H, W): Batch mode for MCTS inference.
        - If cnn_tensor is 5D (1, S, C, H, W): Sequence mode for training.
        """
        # --- MODIFIED: Polymorphic Forward Pass ---
        is_sequence = cnn_tensor.dim() == 5
        if is_sequence:
            # Reshape from [1, S, C, H, W] to [S, C, H, W] for processing
            seq_len = cnn_tensor.size(1)
            cnn_tensor = cnn_tensor.squeeze(0)
        else:
            # This is the MCTS batch case
            seq_len = cnn_tensor.size(0)

        # 1. Process GNN and CNN paths in parallel
        gnn_out = self.gnn(gnn_batch)
        cnn_out = self.cnn(cnn_tensor)

        # 2. Reshape and Pool to get one vector per board state
        gnn_out_reshaped = gnn_out.view(seq_len, 64, self.embed_dim)
        gnn_out_pooled = gnn_out_reshaped.mean(dim=1)

        cnn_out_flat = cnn_out.view(seq_len, self.embed_dim, -1)
        cnn_out_pooled = cnn_out_flat.mean(dim=2)

        # 3. Fuse GNN and CNN embeddings
        fused = torch.cat([gnn_out_pooled, cnn_out_pooled], dim=-1)
        
        # 4. Project fused embedding into the model's main dimension
        final_embedding = self.embedding_projection(fused)

        # 5. Apply Transformer for sequences, otherwise use embedding directly
        if is_sequence:
            # Add a batch dimension for the transformer [S, D] -> [1, S, D]
            final_embedding = self.transformer_encoder(final_embedding.unsqueeze(0))
            # Squeeze back out for the heads [1, S, D] -> [S, D]
            final_embedding = final_embedding.squeeze(0)

        # 6. Get predictions from heads
        policy_logits = self.policy_head(final_embedding)
        value = torch.tanh(self.value_head(final_embedding))
        material_balance = self.material_head(final_embedding)

        return policy_logits, value, material_balance