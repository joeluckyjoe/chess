#
# File: dual_gnn_agent.py
#
"""
This file defines the main neural network agent, which integrates the dual GNNs,
the cross-attention module, and the policy/value heads into a single nn.Module.
"""
import torch
import torch.nn as nn
from typing import Tuple
from gnn_data_converter import GNNInput
from gnn_models import SquareGNN, PieceGNN
from attention_module import CrossAttentionModule # Assuming this is your file name
from policy_value_heads import PolicyHead, ValueHead

class DualGNNAgent(nn.Module):
    """
    The complete MCTS RL agent model, combining dual GNNs, cross-attention,
    and policy/value heads.
    """
    def __init__(self, 
                 gnn_input_dim: int, 
                 gnn_hidden_dim: int, 
                 gnn_embedding_dim: int, 
                 square_gnn_heads: int,
                 n_heads_cross_attention: int,
                 cross_attention_dropout_rate: float = 0.1): # Added dropout for CA
        """
        Initializes the full agent model.

        Args:
            gnn_input_dim (int): The input feature dimension for both GNNs (e.g., 12).
            gnn_hidden_dim (int): The hidden dimension for the GNNs.
            gnn_embedding_dim (int): The output embedding dimension for the GNNs.
                                     This will also be the embedding dimension
                                     for the cross-attention module's input and
                                     the policy/value heads' input.
            square_gnn_heads (int): Number of attention heads for the SquareGNN's GAT layers.
            n_heads_cross_attention (int): The number of attention heads for the
                                           cross-attention module.
            cross_attention_dropout_rate (float): Dropout rate for the CrossAttentionModule.
        """
        super(DualGNNAgent, self).__init__()
        
        # Initialize SquareGNN
        self.square_gnn = SquareGNN(
            in_features=gnn_input_dim,
            hidden_features=gnn_hidden_dim,
            out_features=gnn_embedding_dim,
            heads=square_gnn_heads
        )
        
        # Initialize PieceGNN
        self.piece_gnn = PieceGNN(
            in_channels=gnn_input_dim,
            hidden_channels=gnn_hidden_dim,
            out_channels=gnn_embedding_dim
        )
        
        # Initialize CrossAttentionModule
        self.cross_attention = CrossAttentionModule(
            sq_embed_dim=gnn_embedding_dim, # Square embeddings are queries
            pc_embed_dim=gnn_embedding_dim, # Piece embeddings are keys/values
            num_heads=n_heads_cross_attention,
            dropout_rate=cross_attention_dropout_rate
        )
        
        # Define the Policy and Value Heads
        self.policy_head = PolicyHead(embedding_dim=gnn_embedding_dim)
        self.value_head = ValueHead(embedding_dim=gnn_embedding_dim)

    def forward(self, gnn_input: GNNInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the full forward pass of the network.

        Args:
            gnn_input (GNNInput): A dataclass containing the graph data for
                                  both the square and piece graphs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - policy_logits (torch.Tensor): The raw output from the policy head.
                - value (torch.Tensor): The estimated game outcome from the value head.
        """
        square_graph = gnn_input.square_graph
        piece_graph = gnn_input.piece_graph

        # 1. Process inputs through their respective GNNs
        square_embeddings = self.square_gnn(square_graph.x, square_graph.edge_index)
        # square_embeddings shape: (num_squares, gnn_embedding_dim) -> (64, D)
        
        if piece_graph.x is not None and piece_graph.x.size(0) > 0:
            piece_embeddings = self.piece_gnn(piece_graph.x, piece_graph.edge_index)
            # piece_embeddings shape: (num_pieces, gnn_embedding_dim)
        else:
            device = square_embeddings.device
            piece_embeddings = torch.empty((0, self.piece_gnn.conv2.out_channels), device=device)
            # piece_embeddings shape: (0, D)

        # Prepare for CrossAttentionModule (expects seq_len, batch_size, embed_dim due to batch_first=False)
        # Current batch_size is 1 for this stage of testing.
        # square_embeddings: (64, D) -> unsqueeze for batch -> (64, 1, D)
        query_for_attn = square_embeddings.unsqueeze(1) 

        # piece_embeddings: (num_pieces or 0, D) -> unsqueeze for batch -> (num_pieces or 0, 1, D)
        key_value_for_attn = piece_embeddings.unsqueeze(1)

        # 2. Fuse embeddings with cross-attention
        # CrossAttentionModule expects:
        #   square_embeddings: (num_squares, batch_size, sq_embed_dim)
        #   piece_embeddings: (num_current_pieces, batch_size, pc_embed_dim)
        # Output processed_attended_squares_permuted: (num_squares, batch_size, sq_embed_dim) -> (64, 1, D)
        processed_attended_squares_permuted = self.cross_attention(
            square_embeddings=query_for_attn,
            piece_embeddings=key_value_for_attn
            # piece_padding_mask is not used for now, defaults to None
        )
        
        # Reshape for Policy/Value Heads (they expect batch_size, seq_len, embed_dim)
        # (64, 1, D) -> permute -> (1, 64, D)
        processed_attended_squares = processed_attended_squares_permuted.permute(1, 0, 2)
        
        # 3. Pass the fused representation to the policy and value heads
        policy_logits = self.policy_head(processed_attended_squares)
        value = self.value_head(processed_attended_squares)
        
        return policy_logits, value