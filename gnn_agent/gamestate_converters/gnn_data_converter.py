#
# File: gnn_data_converter.py (Corrected)
#
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import chess
import numpy as np

# --- Constants for Feature Engineering ---
# NOTE: The feature dimension is determined by the feature engineering below.
# It is 2 (pos) + 6 (type) + 2 (color) + 2 (control) = 12 for squares
# and 6 (type) + 1 (color) + 2 (loc) + 1 (mob) + 2 (atk/def) = 12 for pieces.
GNN_INPUT_FEATURE_DIM = 12

# Mapping piece types to an index for one-hot encoding
PIECE_TYPE_MAP: Dict[chess.PieceType, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
NUM_PIECE_TYPES = len(PIECE_TYPE_MAP)  # 6

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
        """Allows unpacking the object like a tuple for the network forward pass."""
        yield self.square_graph.x
        yield self.square_graph.edge_index
        yield self.piece_graph.x
        yield self.piece_graph.edge_index
        yield self.piece_to_square_map

# --- Helper Functions ---

def _create_square_adjacency_edges() -> torch.Tensor:
    """
    Creates a static edge index for the 64 squares of the board.
    An edge exists between any two squares that are adjacent (including diagonals).
    """
    edges = []
    for i in range(64):
        rank, file = i // 8, i % 8
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                new_rank, new_file = rank + dr, file + df
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    neighbor_sq = new_rank * 8 + new_file
                    edges.append((i, neighbor_sq))
    # Transpose to get shape (2, num_edges)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# Pre-calculate the static square graph edges
_SQUARE_ADJACENCY_EDGE_INDEX = _create_square_adjacency_edges()


# --- Main Conversion Function (Corrected) ---

def convert_to_gnn_input(board: chess.Board, device) -> GNNInput:
    """
    Converts a python-chess board state into the GNNInput format.
    This version returns ONLY the GNNInput object for the training loop.
    """
    # 1. Square-based Graph (G_sq)
    square_features_list = []
    for sq in chess.SQUARES:
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
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
        square_features_list.append(
            np.concatenate([pos_encoding, piece_type_one_hot, piece_color_one_hot, control_status])
        )

    square_graph = GNNGraph(
        x=torch.from_numpy(np.array(square_features_list, dtype=np.float32)).to(device),
        edge_index=_SQUARE_ADJACENCY_EDGE_INDEX.to(device)
    )

    # 2. Piece-based Graph (G_pc)
    piece_map = board.piece_map()

    # Handle empty board case for piece graph
    if not piece_map:
        piece_graph = GNNGraph(
            x=torch.empty((0, GNN_INPUT_FEATURE_DIM), dtype=torch.float32, device=device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device)
        )
        piece_to_square_map = torch.empty((0), dtype=torch.long, device=device)
    else:
        piece_node_indices = {sq: i for i, sq in enumerate(piece_map.keys())}
        square_indices_for_pieces = list(piece_map.keys())
        piece_to_square_map = torch.tensor(square_indices_for_pieces, dtype=torch.long, device=device)

        piece_features_list = []
        piece_edges = []

        legal_moves = list(board.legal_moves)
        piece_mobilities = {sq: 0 for sq in piece_map.keys()}
        for move in legal_moves:
            if move.from_square in piece_mobilities:
                piece_mobilities[move.from_square] += 1

        for from_sq, piece in piece_map.items():
            piece_type_one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
            piece_type_one_hot[PIECE_TYPE_MAP[piece.piece_type]] = 1.0
            piece_color = [1.0 if piece.color == chess.WHITE else 0.0]
            rank, file = chess.square_rank(from_sq), chess.square_file(from_sq)
            location = [file / 7.0, rank / 7.0]
            mobility = [piece_mobilities.get(from_sq, 0) / 28.0]
            attack_count = len(board.attacks(from_sq) & board.occupied_co[not piece.color])
            defense_count = len(board.attackers(piece.color, from_sq))
            attack_defense = [float(attack_count), float(defense_count)]

            piece_features_list.append(
                np.concatenate([piece_type_one_hot, piece_color, location, mobility, attack_defense])
            )

            for to_sq in board.attacks(from_sq):
                if to_sq in piece_node_indices:
                    from_node_idx = piece_node_indices[from_sq]
                    to_node_idx = piece_node_indices[to_sq]
                    piece_edges.append((from_node_idx, to_node_idx))

        if not piece_edges:
            piece_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            piece_edge_index = torch.tensor(piece_edges, dtype=torch.long, device=device).t().contiguous()

        piece_graph = GNNGraph(
            x=torch.from_numpy(np.array(piece_features_list, dtype=np.float32)).to(device),
            edge_index=piece_edge_index
        )

    # Corrected return statement
    return GNNInput(
        square_graph=square_graph,
        piece_graph=piece_graph,
        piece_to_square_map=piece_to_square_map
    )