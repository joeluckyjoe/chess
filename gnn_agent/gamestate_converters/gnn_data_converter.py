#
# File: gnn_data_converter.py (Corrected)
#
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import chess
import numpy as np

# --- Constants for Feature Engineering ---
GNN_INPUT_FEATURE_DIM = 12

PIECE_TYPE_MAP: Dict[chess.PieceType, int] = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}
NUM_PIECE_TYPES = len(PIECE_TYPE_MAP)

# --- Data Structures ---

@dataclass
class GNNGraph:
    """Represents the structure of a single graph for the GNN."""
    x: torch.Tensor
    edge_index: torch.Tensor

@dataclass
class GNNInput:
    """The complete input for our dual-GNN model for a single board state."""
    square_graph: GNNGraph
    piece_graph: GNNGraph
    piece_to_square_map: torch.Tensor

    def __iter__(self):
        yield self.square_graph.x
        yield self.square_graph.edge_index
        yield self.piece_graph.x
        yield self.piece_graph.edge_index
        yield self.piece_to_square_map

@dataclass
class BatchedGNNInput:
    """The batched input for our dual-GNN model."""
    square_features: torch.Tensor
    square_edge_index: torch.Tensor
    square_batch: torch.Tensor
    piece_features: torch.Tensor
    piece_edge_index: torch.Tensor
    piece_batch: torch.Tensor
    piece_to_square_map: torch.Tensor
    piece_padding_mask: torch.Tensor

# --- Helper Functions ---

def _create_square_adjacency_edges() -> torch.Tensor:
    """Creates a static edge index for the 64 squares of the board."""
    edges = []
    for i in range(64):
        rank, file = i // 8, i % 8
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0: continue
                new_rank, new_file = rank + dr, file + df
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    neighbor_sq = new_rank * 8 + new_file
                    edges.append((i, neighbor_sq))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

_SQUARE_ADJACENCY_EDGE_INDEX = _create_square_adjacency_edges()

# --- Main Conversion Functions ---

def convert_to_gnn_input(board: chess.Board, device: torch.device) -> GNNInput:
    """Converts a single python-chess board state into the GNNInput format."""
    # 1. Square-based Graph (G_sq)
    square_features_list = []
    for sq in chess.SQUARES:
        rank, file = chess.square_rank(sq), chess.square_file(sq)
        pos_encoding = [file / 7.0, rank / 7.0]
        piece = board.piece_at(sq)
        piece_type_one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
        piece_color_one_hot = np.zeros(2, dtype=np.float32)
        if piece:
            piece_type_one_hot[PIECE_TYPE_MAP[piece.piece_type]] = 1.0
            piece_color_one_hot[0 if piece.color == chess.WHITE else 1] = 1.0
        is_attacked_by_white = float(board.is_attacked_by(chess.WHITE, sq))
        is_attacked_by_black = float(board.is_attacked_by(chess.BLACK, sq))
        control_status = [is_attacked_by_white, is_attacked_by_black]
        square_features_list.append(np.concatenate([pos_encoding, piece_type_one_hot, piece_color_one_hot, control_status]))

    square_graph = GNNGraph(
        x=torch.from_numpy(np.array(square_features_list, dtype=np.float32)),
        edge_index=_SQUARE_ADJACENCY_EDGE_INDEX.clone() # Clone to avoid modifying the original
    )

    # 2. Piece-based Graph (G_pc)
    piece_map = board.piece_map()
    if not piece_map:
        piece_graph = GNNGraph(x=torch.empty((0, GNN_INPUT_FEATURE_DIM)), edge_index=torch.empty((2, 0), dtype=torch.long))
        piece_to_square_map = torch.empty((0), dtype=torch.long)
    else:
        sorted_squares = sorted(piece_map.keys())
        piece_node_indices = {sq: i for i, sq in enumerate(sorted_squares)}
        piece_to_square_map = torch.tensor(sorted_squares, dtype=torch.long)
        
        piece_features_list, piece_edges = [], []
        piece_mobilities = {sq: 0 for sq in sorted_squares}
        for move in board.legal_moves:
            if move.from_square in piece_mobilities:
                piece_mobilities[move.from_square] += 1
        
        for from_sq in sorted_squares:
            piece = board.piece_at(from_sq)
            piece_type_one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
            piece_type_one_hot[PIECE_TYPE_MAP[piece.piece_type]] = 1.0
            piece_color = [1.0 if piece.color == chess.WHITE else 0.0]
            rank, file = chess.square_rank(from_sq), chess.square_file(from_sq)
            location = [file / 7.0, rank / 7.0]
            mobility = [piece_mobilities.get(from_sq, 0) / 28.0]
            attack_count = len(board.attacks(from_sq) & board.occupied_co[not piece.color])
            defense_count = len(board.attackers(piece.color, from_sq))
            attack_defense = [float(attack_count), float(defense_count)]
            piece_features_list.append(np.concatenate([piece_type_one_hot, piece_color, location, mobility, attack_defense]))
            
            for to_sq in board.attacks(from_sq):
                if to_sq in piece_node_indices:
                    piece_edges.append((piece_node_indices[from_sq], piece_node_indices[to_sq]))
        
        piece_edge_index = torch.tensor(piece_edges, dtype=torch.long).t().contiguous() if piece_edges else torch.empty((2, 0), dtype=torch.long)
        piece_graph = GNNGraph(
            x=torch.from_numpy(np.array(piece_features_list, dtype=np.float32)),
            edge_index=piece_edge_index
        )

    return GNNInput(
        square_graph=square_graph,
        piece_graph=piece_graph,
        piece_to_square_map=piece_to_square_map
    )


def convert_boards_to_gnn_batch(boards: List[chess.Board], device: torch.device) -> BatchedGNNInput:
    """
    Converts a list of python-chess boards into a single batched GNNInput object.
    This is the primary function for batch processing.
    """
    gnn_inputs = [convert_to_gnn_input(b, torch.device('cpu')) for b in boards]

    # --- Collate Lists of Tensors ---
    square_x_list = [g.square_graph.x for g in gnn_inputs]
    piece_x_list = [g.piece_graph.x for g in gnn_inputs]
    square_edge_indices = [g.square_graph.edge_index for g in gnn_inputs]
    piece_edge_indices = [g.piece_graph.edge_index for g in gnn_inputs]
    piece_to_square_maps = [g.piece_to_square_map for g in gnn_inputs]

    # --- Create Batch Tensors for PyG ---
    square_batch = torch.tensor([i for i, x in enumerate(square_x_list) for _ in range(x.size(0))], dtype=torch.long)
    piece_batch = torch.tensor([i for i, x in enumerate(piece_x_list) for _ in range(x.size(0))], dtype=torch.long)

    # --- Stack Tensors and Adjust Edge Indices ---
    square_features = torch.cat(square_x_list, dim=0)
    piece_features = torch.cat(piece_x_list, dim=0)
    
    csum_sq = torch.cumsum(torch.tensor([g.square_graph.x.size(0) for g in gnn_inputs]), 0)
    csum_sq = torch.cat([torch.tensor([0]), csum_sq[:-1]])
    square_edge_index = torch.cat([e + c for e, c in zip(square_edge_indices, csum_sq)], dim=1)

    csum_pc = torch.cumsum(torch.tensor([g.piece_graph.x.size(0) for g in gnn_inputs]), 0)
    csum_pc = torch.cat([torch.tensor([0]), csum_pc[:-1]])
    piece_edge_index = torch.cat([e + c for e, c in zip(piece_edge_indices, csum_pc)], dim=1)

    # The piece_to_square_map needs to be adjusted for the batched square graph
    adjusted_piece_to_square_map = torch.cat([p_map + c_sq for p_map, c_sq in zip(piece_to_square_maps, csum_sq)], dim=0)
    
    # --- Create Padding Mask for Attention ---
    max_pieces = max(x.size(0) for x in piece_x_list) if piece_x_list else 0
    batch_size = len(boards)
    piece_padding_mask = torch.ones((batch_size, max_pieces), dtype=torch.bool)
    if piece_x_list:
        for i, x in enumerate(piece_x_list):
            piece_padding_mask[i, :x.size(0)] = 0 # 0 means not masked

    return BatchedGNNInput(
        square_features=square_features.to(device),
        square_edge_index=square_edge_index.to(device),
        square_batch=square_batch.to(device),
        piece_features=piece_features.to(device),
        piece_edge_index=piece_edge_index.to(device),
        piece_batch=piece_batch.to(device),
        piece_to_square_map=adjusted_piece_to_square_map.to(device),
        piece_padding_mask=piece_padding_mask.to(device)
    )