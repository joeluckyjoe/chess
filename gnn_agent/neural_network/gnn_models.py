# gnn_agent/neural_network/gnn_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_max_pool
from typing import Optional, Tuple

# --- Constants from gnn_data_converter ---
from gnn_agent.gamestate_converters.gnn_data_converter import SQUARE_FEATURE_DIM, PIECE_FEATURE_DIM

# --- GNN Modules ---

class SquareGNN(nn.Module):
    """A 2-layer Graph Attention Network for processing the 64 squares."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int, heads: int = 4):
        super(SquareGNN, self).__init__()
        self.conv1 = GATv2Conv(in_features, hidden_features, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_features * heads, out_features, heads=1, concat=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        return x

class PieceGNN(nn.Module):
    """A 3-layer Graph Convolutional Network for processing piece relationships."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(PieceGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x_piece: torch.Tensor, edge_index_piece: torch.Tensor) -> torch.Tensor:
        if x_piece is None or x_piece.size(0) == 0:
            return torch.empty((0, self.conv3.out_channels), device=edge_index_piece.device)
        x = self.conv1(x_piece, edge_index_piece)
        x = F.gelu(x)
        x = self.conv2(x, edge_index_piece)
        x = F.gelu(x)
        x = self.conv3(x, edge_index_piece)
        return x

class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.piece_to_square_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.square_to_piece_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn_piece = nn.Linear(embed_dim, embed_dim)
        self.ffn_square = nn.Linear(embed_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, square_features: torch.Tensor, piece_features: torch.Tensor, piece_to_square_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        key_value = square_features[piece_to_square_map]
        query = piece_features
        query_r, key_value_r = query.unsqueeze(0), key_value.unsqueeze(0)
        attended_pieces, _ = self.piece_to_square_attention(query_r, key_value_r, key_value_r)
        attended_pieces = self.norm1(attended_pieces.squeeze(0) + piece_features)
        attended_pieces_ffn = self.ffn_piece(attended_pieces)
        attended_pieces = self.norm3(F.gelu(attended_pieces_ffn) + attended_pieces)
        global_piece_context = torch.max(piece_features, dim=0, keepdim=True)[0]
        global_piece_context_r = global_piece_context.unsqueeze(0)
        square_features_r = square_features.unsqueeze(0)
        attended_squares, _ = self.square_to_piece_attention(square_features_r, global_piece_context_r, global_piece_context_r)
        attended_squares = self.norm2(attended_squares.squeeze(0) + square_features)
        attended_squares_ffn = self.ffn_square(attended_squares)
        attended_squares = self.norm4(F.gelu(attended_squares_ffn) + attended_squares)
        return attended_squares, attended_pieces

class PolicyHead(nn.Module):
    def __init__(self, embed_dim: int, action_space_size: int = 4672):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class ValueHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return torch.tanh(x)


# --- Main Network ---

class ChessNetwork(nn.Module):
    """The main network orchestrating GNNs, attention, and heads."""
    def __init__(self, embed_dim: int = 256, gnn_hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.square_feature_dim = SQUARE_FEATURE_DIM
        self.piece_feature_dim = PIECE_FEATURE_DIM
        
        self.square_gnn = SquareGNN(SQUARE_FEATURE_DIM, gnn_hidden_dim, embed_dim, heads=num_heads)
        self.piece_gnn = PieceGNN(PIECE_FEATURE_DIM, embed_dim, embed_dim)
        self.fusion = CrossAttentionModule(embed_dim, num_heads)
        self.policy_head = PolicyHead(embed_dim)
        self.value_head = ValueHead(embed_dim)

    # This forward signature now correctly matches the keyword arguments passed by mcts.py
    def forward(self, square_features, square_edge_index, square_batch,
                piece_features, piece_edge_index, piece_batch,
                piece_to_square_map, piece_padding_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # The 'batch' argument is not passed to the GNNs directly.
        sq_embed = self.square_gnn(square_features, square_edge_index)
        pc_embed = self.piece_gnn(piece_features, piece_edge_index)

        if pc_embed.size(0) > 0:
            sq_embed_fused, pc_embed_fused = self.fusion(sq_embed, pc_embed, piece_to_square_map)
            
            batch_size = square_batch.max().item() + 1
            global_graph_embed = global_max_pool(
                pc_embed_fused, 
                piece_batch.to(pc_embed_fused.device),
                size=batch_size
            )
        else: # Handle boards with no pieces
            global_graph_embed = global_max_pool(sq_embed, square_batch)

        policy_logits = self.policy_head(global_graph_embed)
        value_estimate = self.value_head(global_graph_embed)

        return policy_logits, value_estimate.squeeze(-1)