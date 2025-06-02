#
# File: gnn_data_converter.py (Refactored)
#
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import chess
import numpy as np

def get_move_to_index_map():
    """
    Creates a comprehensive mapping from every possible chess move to a unique integer index.
    This includes regular moves and all possible promotion pieces.

    Returns:
        dict: A dictionary mapping chess.Move objects to integer indices.
    """
    move_to_index = {}
    index = 0
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            # Handle regular moves
            move = chess.Move(from_square, to_square)
            if move not in move_to_index:
                move_to_index[move] = index
                index += 1
            
            # Handle promotions
            for promotion_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_square, to_square, promotion=promotion_piece)
                if promo_move not in move_to_index:
                    move_to_index[promo_move] = index
                    index += 1
    return move_to_index

# You can also add this for convenience, so other modules can import it directly
MOVE_TO_INDEX_MAP = get_move_to_index_map()

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


# --- Main Conversion Function (Refactored) ---

def convert_to_gnn_input(
    board: chess.Board,
    device: torch.device,
    for_visualization: bool = False
) -> Union[GNNInput, Tuple[GNNInput, List[str]]]:
    """
    Converts a python-chess board state into the GNNInput format.

    Args:
        board: The python-chess board object.
        device: The torch device to move tensors to.
        for_visualization: If True, also returns labels for plotting.

    Returns:
        If for_visualization is False, returns the GNNInput object.
        If for_visualization is True, returns a tuple containing:
            - The GNNInput object.
            - A list of strings representing piece labels for visualization.
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
    piece_labels_for_plot = []

    # Handle empty board case for piece graph
    if not piece_map:
        piece_graph = GNNGraph(
            x=torch.empty((0, GNN_INPUT_FEATURE_DIM), dtype=torch.float32, device=device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device)
        )
        piece_to_square_map = torch.empty((0), dtype=torch.long, device=device)
    else:
        # Important: Sort keys to ensure consistent node ordering
        sorted_squares = sorted(piece_map.keys())
        piece_node_indices = {sq: i for i, sq in enumerate(sorted_squares)}
        piece_to_square_map = torch.tensor(sorted_squares, dtype=torch.long, device=device)

        piece_features_list = []
        piece_edges = []

        legal_moves = list(board.legal_moves)
        piece_mobilities = {sq: 0 for sq in sorted_squares}
        for move in legal_moves:
            if move.from_square in piece_mobilities:
                piece_mobilities[move.from_square] += 1

        for from_sq in sorted_squares:
            piece = board.piece_at(from_sq)

            # Generate labels for visualization if requested
            if for_visualization:
                color_str = 'w' if piece.color == chess.WHITE else 'b'
                piece_str = piece.symbol().upper()
                piece_labels_for_plot.append(f"{color_str}{piece_str}")

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

    gnn_input = GNNInput(
        square_graph=square_graph,
        piece_graph=piece_graph,
        piece_to_square_map=piece_to_square_map
    )

    if for_visualization:
        return gnn_input, piece_labels_for_plot
    else:
        return gnn_input