#
# File: gnn_agent/neural_network/gnn_models.py (Corrected PolicyHead)
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_max_pool
from typing import Optional, Tuple

from gnn_agent.gamestate_converters.gnn_data_converter import SQUARE_FEATURE_DIM, PIECE_FEATURE_DIM

# --- LEGACY GNN Modules (No longer used by the new ChessNetwork) ---
class SquareGNN(nn.Module):
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
        self.norm1, self.norm2, self.norm3, self.norm4 = [nn.LayerNorm(embed_dim) for _ in range(4)]
        self.ffn_piece = nn.Linear(embed_dim, embed_dim)
        self.ffn_square = nn.Linear(embed_dim, embed_dim)
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

# --- Current Head Modules ---

class PolicyHead(nn.Module):
    def __init__(self, trunk_dim: int, action_space_size: int = 4672):
        super().__init__()
        self.fc1 = nn.Linear(trunk_dim, 1024)
        self.fc2 = nn.Linear(1024, action_space_size)
        self.ln1 = nn.LayerNorm(1024)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        # --- BUG FIX: Removed F.log_softmax(x, dim=-1) ---
        # F.cross_entropy in the trainer applies softmax internally.
        # Applying it here was incorrect and created a flawed gradient signal.
        return x

class ValueHead(nn.Module):
    def __init__(self, trunk_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(trunk_dim, 512)
        self.fc2 = nn.Linear(512, 1)
        self.ln1 = nn.LayerNorm(512)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        return torch.tanh(x)