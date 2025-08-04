import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from .policy_value_model import PolicyValueModel
from gnn_agent.gamestate_converters.action_space_converter import get_action_space_size

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding layer for Transformer models.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 16):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalPolicyValueModel(nn.Module):
    """
    The main model for Phase D, combining a GNN+CNN encoder with a Transformer
    to process sequences of board states.
    """
    # <<< MODIFIED: Added fine_tune_encoder argument to the constructor.
    def __init__(self, encoder_model: PolicyValueModel, policy_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1, fine_tune_encoder: bool = False):
        super().__init__()
        self.d_model = d_model

        # 1. The GNN+CNN Encoder
        self.encoder = encoder_model
        # <<< MODIFIED: Encoder freezing is now conditional.
        if not fine_tune_encoder:
            print("INFO: Encoder parameters are FROZEN.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("INFO: Encoder parameters are UNFROZEN for fine-tuning.")


        # 2. State Sequencer's Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Temporal Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # We use (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_layers
        )

        # 4. Policy and Value Heads (Correctly Sized)
        self.policy_head = nn.Linear(d_model, policy_size)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, gnn_batch: Batch, cnn_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the temporal model.
        """
        batch_size, seq_len = cnn_batch.shape[0], cnn_batch.shape[1]

        # <<< MODIFIED: Removed the unconditional self.encoder.eval().
        # The training script will now manage the mode (train/eval) of the entire model.

        # Reshape CNN tensor for batch processing through the encoder
        cnn_input = cnn_batch.view(batch_size * seq_len, *cnn_batch.shape[2:])

        # Pass the entire batch of GNN and CNN data through the encoder at once.
        _, _, embeddings = self.encoder(gnn_batch, cnn_input)
        
        # Reshape embeddings for the Transformer
        seq_embeddings = embeddings.view(batch_size, seq_len, self.d_model)
        seq_embeddings = seq_embeddings.permute(1, 0, 2)

        # Add positional encoding
        seq_embeddings_with_pos = self.pos_encoder(seq_embeddings)

        # Pass through the Transformer encoder
        transformer_output = self.transformer_encoder(seq_embeddings_with_pos)

        # Use the output of the *last* token in the sequence for the final prediction.
        final_embedding = transformer_output[-1, :, :]
        
        # Final predictions
        policy_logits = self.policy_head(final_embedding)
        value = self.value_head(final_embedding)

        return policy_logits, torch.tanh(value)