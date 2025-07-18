import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import Tuple

from .unified_gnn import UnifiedGNN
from .cnn_model import CNNModel

class HybridTransformerModel(nn.Module):
    """
    Implements Phase BK: GNN+CNN+Transformer.

    This model fuses relational (GNN) and spatial (CNN) embeddings, then
    processes the resulting sequence of board representations through a
    Transformer Encoder. This allows the model to apply self-attention across
    the entire game sequence, enabling a more holistic understanding of
    temporal dynamics and strategic context.

    It outputs policy, value, and material balance predictions for each
    position in the sequence.
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
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, embed_dim), # Project fused GNN+CNN back to a consistent dimension
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        # --- Transformer Encoder for Temporal Fusion ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            batch_first=True, # Expects (batch, seq, feature)
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)

        # --- Output Heads ---
        self.policy_head = nn.Linear(embed_dim, policy_size)
        self.value_head = nn.Linear(embed_dim, 1)
        self.material_head = nn.Linear(embed_dim, 1)

    def forward(self, gnn_batch: Batch, cnn_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes a sequence of board states.

        Args:
            gnn_batch (Batch): A PyG Batch object containing graph data for the ENTIRE sequence.
            cnn_tensor (torch.Tensor): A tensor for the CNN of shape (sequence_length, channels, 8, 8).

        Returns:
            A tuple containing tensors for the entire sequence:
            - policy_logits (torch.Tensor): (sequence_length, policy_size)
            - value (torch.Tensor): (sequence_length, 1)
            - material_balance (torch.Tensor): (sequence_length, 1)
        """
        # 1. Process GNN and CNN paths in parallel
        # gnn_out: [total_squares_in_sequence, embed_dim]
        # cnn_out: [sequence_length, embed_dim, 8, 8]
        gnn_out = self.gnn(gnn_batch)
        cnn_out = self.cnn(cnn_tensor)

        # 2. Reshape and Pool to get one vector per board state
        seq_len = cnn_tensor.size(0)

        # Pool GNN output per board state
        gnn_out_reshaped = gnn_out.view(seq_len, 64, self.embed_dim)
        gnn_out_pooled = gnn_out_reshaped.mean(dim=1) # [seq_len, embed_dim]

        # Pool CNN output per board state
        cnn_out_flat = cnn_out.view(seq_len, self.embed_dim, -1)
        cnn_out_pooled = cnn_out_flat.mean(dim=2) # [seq_len, embed_dim]

        # 3. Fuse GNN and CNN embeddings
        fused = torch.cat([gnn_out_pooled, cnn_out_pooled], dim=-1) # [seq_len, 2 * embed_dim]
        
        # 4. Project fused embedding into the transformer's expected dimension
        # The unsqueeze(0) adds the batch dimension (we process one game sequence at a time)
        fused_projected = self.fusion_mlp(fused).unsqueeze(0) # [1, seq_len, embed_dim]

        # 5. Pass the entire sequence through the Transformer Encoder
        transformer_output = self.transformer_encoder(fused_projected) # [1, seq_len, embed_dim]
        
        # Remove the batch dimension for the heads
        transformer_output_flat = transformer_output.squeeze(0) # [seq_len, embed_dim]

        # 6. Get predictions from all three heads for each step in the sequence
        policy_logits = self.policy_head(transformer_output_flat)
        value = torch.tanh(self.value_head(transformer_output_flat))
        material_balance = self.material_head(transformer_output_flat)

        return policy_logits, value, material_balance