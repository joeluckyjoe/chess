# gnn_data_converter.py

from dataclasses import dataclass
from typing import Dict, List, Tuple

import chess
import numpy as np

# --- Constants for Feature Engineering ---

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
    node_features: np.ndarray  # Shape: (num_nodes, num_node_features)
    edge_index: np.ndarray      # Shape: (2, num_edges), COO format for sparse graphs

@dataclass
class GNNInput:
    """The complete input for our dual-GNN model for a single board state."""
    square_graph: GNNGraph
    piece_graph: GNNGraph

# --- Helper Functions ---

def _create_square_adjacency_edges() -> np.ndarray:
    """
    Creates a static edge index for the 64 squares of the board.
    An edge exists between any two squares that are adjacent (including diagonals).
    """
    edges = []
    for i in range(64):
        # Can't use python-chess for this as it requires a piece on the square
        # Manually calculate neighbors
        rank, file = i // 8, i % 8
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                new_rank, new_file = rank + dr, file + df
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    neighbor_sq = new_rank * 8 + new_file
                    edges.append((i, neighbor_sq))
    return np.array(edges, dtype=np.int64).T

# Pre-calculate the static square graph edges
_SQUARE_ADJACENCY_EDGE_INDEX = _create_square_adjacency_edges()


# --- Main Conversion Function ---

def convert_to_gnn_input(board: chess.Board) -> GNNInput:
    """
    Converts a python-chess board state into the GNNInput format.

    Args:
        board: The current python-chess board object.

    Returns:
        A GNNInput object containing feature and graph data for both
        the square-based and piece-based GNNs.
    """
    # 1. Square-based Graph (G_sq)
    square_features_list = []
    for sq in chess.SQUARES:
        # NOTE: For clarity and robustness, I've slightly simplified the square
        # 'attack/defense' features from the original proposal to be a direct
        # representation of control, which is less ambiguous. The total feature
        # count per square is now 12.
        
        # Positional encoding (2 features)
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        pos_encoding = [file / 7.0, rank / 7.0]

        # Piece type and color (6 + 2 features)
        piece = board.piece_at(sq)
        piece_type_one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
        piece_color_one_hot = np.zeros(2, dtype=np.float32) # [white, black]
        if piece:
            piece_type_one_hot[PIECE_TYPE_MAP[piece.piece_type]] = 1.0
            piece_color_one_hot[0 if piece.color == chess.WHITE else 1] = 1.0
        
        # Board control status (2 features)
        is_attacked_by_white = float(board.is_attacked_by(chess.WHITE, sq))
        is_attacked_by_black = float(board.is_attacked_by(chess.BLACK, sq))
        control_status = [is_attacked_by_white, is_attacked_by_black]

        # Combine all features for the square
        square_features_list.append(
            np.concatenate([pos_encoding, piece_type_one_hot, piece_color_one_hot, control_status])
        )

    square_graph = GNNGraph(
        node_features=np.array(square_features_list, dtype=np.float32),
        edge_index=_SQUARE_ADJACENCY_EDGE_INDEX
    )

    # 2. Piece-based Graph (G_pc)
    piece_map = board.piece_map()
    piece_node_indices = {sq: i for i, sq in enumerate(piece_map.keys())}
    num_pieces = len(piece_node_indices)
    
    piece_features_list = []
    piece_edges = []
    
    # Calculate piece mobilities once
    legal_moves = list(board.legal_moves)
    piece_mobilities = {sq: 0 for sq in piece_map.keys()}
    for move in legal_moves:
        if move.from_square in piece_mobilities:
            piece_mobilities[move.from_square] += 1

    for from_sq, piece in piece_map.items():
        # Piece Type (6 features)
        piece_type_one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
        piece_type_one_hot[PIECE_TYPE_MAP[piece.piece_type]] = 1.0
        
        # Piece Color (1 feature)
        piece_color = [1.0 if piece.color == chess.WHITE else 0.0]

        # Location (2 features)
        rank = chess.square_rank(from_sq)
        file = chess.square_file(from_sq)
        location = [file / 7.0, rank / 7.0]
        
        # Mobility (1 feature) - normalized by a reasonable max (28 for a queen)
        mobility = [piece_mobilities[from_sq] / 28.0]
        
        # Attack/Defense Counts (2 features)
        attack_count = len(board.attacks(from_sq) & board.occupied_co[not piece.color])
        defense_count = len(board.attackers(piece.color, from_sq))
        attack_defense = [float(attack_count), float(defense_count)]
        
        piece_features_list.append(
            np.concatenate([piece_type_one_hot, piece_color, location, mobility, attack_defense])
        )
        
        # Create edges: an edge exists if a piece attacks another's square
        for to_sq in board.attacks(from_sq):
            if to_sq in piece_node_indices: # If the attacked square has a piece
                from_node_idx = piece_node_indices[from_sq]
                to_node_idx = piece_node_indices[to_sq]
                piece_edges.append((from_node_idx, to_node_idx))

    piece_graph = GNNGraph(
        node_features=np.array(piece_features_list, dtype=np.float32),
        edge_index=np.array(piece_edges, dtype=np.int64).T if piece_edges else np.empty((2, 0), dtype=np.int64)
    )

    return GNNInput(square_graph=square_graph, piece_graph=piece_graph)