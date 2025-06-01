# gnn_agent/neural_network/chess_network.py

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from .gnn_models import SquareGNN, PieceGNN
from .attention_module import CrossAttentionModule # Assuming this is the updated one from attention_module_v2
from .policy_value_heads import PolicyHead, ValueHead

class ChessNetwork(nn.Module):
    """
    The main neural network that integrates the dual GNNs, cross-attention,
    and policy/value heads.
    """
    def __init__(self,
                 # SquareGNN params
                 square_in_features: int = 12,
                 square_hidden_features: int = 64,
                 # square_out_features: int = 128, # Effectively overridden by embed_dim
                 square_gat_heads: int = 4,
                 # PieceGNN params
                 piece_in_features: int = 12,
                 piece_hidden_features: int = 32,
                 # piece_out_features: int = 128, # Effectively overridden by embed_dim
                 # Attention params
                 embed_dim: int = 128, # This is the primary dimension for GNN outputs and attention
                 num_heads: int = 4,
                 attention_dropout_rate: float = 0.1, # Added for CrossAttentionModule
                 # Heads params
                 # board_size: int = 64, # Currently unused
                 num_actions: int = 4672):
        super(ChessNetwork, self).__init__()

        self.embed_dim = embed_dim # Store embed_dim for reshaping later if needed

        # Dual GNN encoders
        # Output dimension for both GNNs is set to embed_dim
        self.square_gnn = SquareGNN(
            in_features=square_in_features,
            hidden_features=square_hidden_features,
            out_features=embed_dim, # Using embed_dim here
            heads=square_gat_heads
        )
        self.piece_gnn = PieceGNN(
            in_channels=piece_in_features,
            hidden_channels=piece_hidden_features,
            out_channels=embed_dim # Using embed_dim here
        )

        # Cross-attention module to fuse GNN outputs
        self.cross_attention = CrossAttentionModule(
            sq_embed_dim=embed_dim, 
            pc_embed_dim=embed_dim, 
            num_heads=num_heads,
            dropout_rate=attention_dropout_rate
        )

        # Combined embedding layer after concatenation
        # Input is embed_dim (from square_embeddings) + embed_dim (from attended_square_embeddings)
        self.embedding_layer = nn.Linear(embed_dim * 2, embed_dim) 

        # Policy and Value Heads
        self.policy_head = PolicyHead(embed_dim, num_actions) # Assuming PolicyHead expects (B, D, H, W) or (B, S, D)
        self.value_head = ValueHead(embed_dim)   # Assuming ValueHead expects (B, S, D) or (B, D) after pooling

    def forward(self, 
                square_features: torch.Tensor, 
                square_edge_index: torch.Tensor, 
                piece_features: torch.Tensor, 
                piece_edge_index: torch.Tensor, 
                piece_to_square_map: Optional[torch.Tensor] = None, # Currently unused
                piece_padding_mask: Optional[torch.Tensor] = None, # For attention
                return_attention_weights: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                           Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass for the full network.
        Assumes inputs are for a single graph instance (batch size 1 internally managed for attention).

        Args:
            square_features (Tensor): Node features for all 64 squares. Shape: (num_squares, square_in_features)
            square_edge_index (Tensor): Edge index for the square graph.
            piece_features (Tensor): Node features for the pieces on the board. Shape: (num_pieces, piece_in_features)
            piece_edge_index (Tensor): Edge index for the piece graph.
            piece_to_square_map (Tensor, optional): Maps piece embedding indices to their square indices. Currently unused.
            piece_padding_mask (Tensor, optional): Boolean mask for piece embeddings in attention.
                                                   Shape: (num_pieces) for a single instance, or (batch_size, num_pieces).
                                                   If single instance (num_pieces), it will be unsqueezed.
            return_attention_weights (bool): If True, also returns attention weights.

        Returns:
            If return_attention_weights is False:
                policy_logits (Tensor): Raw output for the policy head. Shape: (num_actions)
                value (Tensor): Scalar value prediction. Shape: (1)
            If return_attention_weights is True:
                policy_logits (Tensor): As above.
                value (Tensor): As above.
                attention_weights (Tensor, optional): Attention weights from the cross-attention module.
                                                      Shape: (num_squares, num_pieces) if B=1 was squeezed.
                                                      None if no pieces or not requested.
        """
        # 1. Process inputs through GNNs
        # Assuming GNNs output shape: (num_nodes, embed_dim) for a single graph
        square_embeddings = self.square_gnn(square_features, square_edge_index) # (64, embed_dim)
        piece_embeddings = self.piece_gnn(piece_features, piece_edge_index)     # (num_pieces, embed_dim)

        actual_attention_weights = None
        
        # Handle case where there are no pieces
        if piece_embeddings.size(0) == 0:
            # If no pieces, the state is primarily defined by square control.
            # To maintain a consistent structure for fusion, we might need a zero tensor for attended part,
            # or adjust self.embedding_layer to handle single embedding_dim input.
            # For now, following the original logic of direct assignment, but this might need refinement
            # if concatenation is strictly enforced by embedding_layer.
            # A simple approach: create a "zero" attended embedding.
            # This assumes embedding_layer expects embed_dim * 2.
            # A more robust way would be to have separate paths or a modified embedding_layer.
            # Sticking to user's original logic: if no pieces, fusion path is skipped.
            # However, the original 'fused_embeddings = square_embeddings' would fail if 
            # self.embedding_layer is always used after concatenation.
            # Let's assume a simplified fusion if no pieces:
            # Option 1: Pass only square_embeddings (requires head to adapt or different processing path)
            # Option 2: Create dummy attended_square_embeddings (e.g., zeros)
            
            # For now, let's make a simple assumption that if no pieces, the "attended" part is zero,
            # and concatenation proceeds. This keeps the structure for embedding_layer.
            # This is a placeholder; a more sophisticated handling might be needed.
            attended_square_embeddings = torch.zeros_like(square_embeddings)

        else:
            # 2. Apply cross-attention
            # Unsqueeze to add a batch dimension of 1 for the attention module
            # square_embeddings: (64, D) -> (64, 1, D)
            # piece_embeddings: (N_pc, D) -> (N_pc, 1, D)
            square_embeddings_b = square_embeddings.unsqueeze(1)
            piece_embeddings_b = piece_embeddings.unsqueeze(1)
            
            # Prepare piece_padding_mask for batch_size=1 if provided
            # CrossAttentionModule expects (batch_size, num_current_pieces)
            piece_padding_mask_b = None
            if piece_padding_mask is not None:
                if piece_padding_mask.dim() == 1: # Shape (num_pieces)
                    piece_padding_mask_b = piece_padding_mask.unsqueeze(0) # Shape (1, num_pieces)
                else: # Already has batch dimension
                    piece_padding_mask_b = piece_padding_mask


            attention_result = self.cross_attention(
                square_embeddings=square_embeddings_b,
                piece_embeddings=piece_embeddings_b,
                piece_padding_mask=piece_padding_mask_b,
                return_attention_weights=return_attention_weights
            )

            if return_attention_weights:
                attended_square_embeddings_b, weights_b = attention_result
                if weights_b is not None:
                    actual_attention_weights = weights_b.squeeze(0) # (1, 64, N_pc) -> (64, N_pc)
                attended_square_embeddings = attended_square_embeddings_b.squeeze(1) # (64, 1, D) -> (64, D)
            else:
                attended_square_embeddings_b = attention_result
                attended_square_embeddings = attended_square_embeddings_b.squeeze(1) # (64, 1, D) -> (64, D)

        # 3. Combine embeddings
        # Concatenate the original square embeddings with the (processed) attended square embeddings
        # Both are (64, embed_dim), so concatenated is (64, embed_dim * 2)
        combined_embeddings = torch.cat([square_embeddings, attended_square_embeddings], dim=1)
        fused_embeddings = self.embedding_layer(combined_embeddings) # (64, embed_dim)

        # 4. Add a dummy batch dimension for the heads, as per original structure.
        # fused_embeddings: (64, embed_dim) -> (1, 64, embed_dim)
        # This implies PolicyHead and ValueHead expect this specific shape for a single instance.
        fused_embeddings_for_heads = fused_embeddings.unsqueeze(0)

        # 5. Pass the batched embeddings to the heads
        # Assuming heads are designed to take (1, 64, embed_dim) and output (1, num_actions) and (1,1)
        policy_logits_b = self.policy_head(fused_embeddings_for_heads)
        value_b = self.value_head(fused_embeddings_for_heads)

        # 6. Remove the dummy batch dimension for the output.
        policy_logits = policy_logits_b.squeeze(0) # (1, num_actions) -> (num_actions)
        value_out = value_b.squeeze(0)             # (1, 1) -> (1)
        
        # Apply tanh to value output as in original
        final_value = torch.tanh(value_out)

        if return_attention_weights:
            return policy_logits, final_value, actual_attention_weights
        else:
            return policy_logits, final_value
