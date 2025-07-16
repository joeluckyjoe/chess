#
# File: gnn_agent/gamestate_converters/gnn_data_converter.py (Updated for Phase BI)
#
import torch
import chess
import numpy as np
from typing import Dict, Tuple
from torch_geometric.data import HeteroData

# --- Constants for Feature Engineering ---
SQUARE_FEATURE_DIM = 13
PIECE_FEATURE_DIM = 22
CNN_INPUT_CHANNELS = 14  # 12 for pieces, 1 for turn, 1 for half-move clock

PIECE_TYPE_MAP: Dict[chess.PieceType, int] = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}
NUM_PIECE_TYPES = len(PIECE_TYPE_MAP)

PIECE_MATERIAL_VALUE: Dict[chess.PieceType, float] = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,
}


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


def _create_cnn_input_tensor(board: chess.Board) -> torch.Tensor:
    """
    Creates the 2D tensor representation for the CNN.
    Shape: (14, 8, 8)
    - 12 channels for piece positions (6 types x 2 colors)
    - 1 channel for player turn (1.0 for White, 0.0 for Black)
    - 1 channel for normalized half-move clock
    """
    tensor = np.zeros((CNN_INPUT_CHANNELS, 8, 8), dtype=np.float32)

    # Channel 0-11: Piece locations
    for sq, piece in board.piece_map().items():
        rank, file = chess.square_rank(sq), chess.square_file(sq)
        piece_idx = PIECE_TYPE_MAP[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else NUM_PIECE_TYPES
        channel = piece_idx + color_offset
        tensor[channel, rank, file] = 1.0

    # Channel 12: Player turn
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    else:
        tensor[12, :, :] = 0.0

    # Channel 13: Half-move clock
    tensor[13, :, :] = board.halfmove_clock / 100.0

    return torch.from_numpy(tensor)


# --- Main Conversion Function ---

def convert_to_gnn_input(board: chess.Board, device: torch.device) -> Tuple[HeteroData, torch.Tensor, torch.Tensor]:
    """
    Converts a single python-chess board state into the required inputs for
    the GNN+CNN hybrid model, plus the ground-truth material balance.

    NOTE: This function's output has changed for Phase BI. It now returns a tuple
    containing (HeteroData_for_GNN, Tensor_for_CNN, Tensor_for_Material_Balance).
    The data loading pipeline must be updated to handle this.

    Returns:
        Tuple[HeteroData, torch.Tensor, torch.Tensor]: A tuple containing:
            - The HeteroData object for the GNN.
            - The (14, 8, 8) tensor for the CNN.
            - A (1,) tensor for the ground-truth material balance.
    """
    # --- 1. Create CNN Input Tensor ---
    cnn_tensor = _create_cnn_input_tensor(board)

    # --- 2. Global Game State Features (for GNN) & Material Balance ---
    turn = 1.0 if board.turn == chess.WHITE else 0.0
    can_castle_wk = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    can_castle_wq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    can_castle_bk = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    can_castle_bq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    ep_square = board.ep_square
    en_passant_file = (chess.square_file(ep_square) / 7.0) if ep_square is not None else 0.0
    halfmove_clock = board.halfmove_clock / 100.0

    white_material = sum(PIECE_MATERIAL_VALUE[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_material = sum(PIECE_MATERIAL_VALUE[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    raw_balance = white_material - black_material
    perspective_balance = raw_balance if board.turn == chess.WHITE else -raw_balance
    
    # Create the separate material balance tensor for the auxiliary loss
    material_balance_tensor = torch.tensor([perspective_balance], dtype=torch.float32)

    # Continue with GNN features
    normalized_balance = perspective_balance / 39.0
    repetition_counter = 1.0 if board.can_claim_threefold_repetition() else 0.0

    global_features = np.array([
        turn, can_castle_wk, can_castle_wq, can_castle_bk, can_castle_bq,
        en_passant_file, halfmove_clock, normalized_balance, repetition_counter
    ], dtype=np.float32)

    # --- 3. Initialize HeteroData Object ---
    data = HeteroData()

    # --- 4. Node Features (for GNN) ---
    square_features_list = []
    for sq in chess.SQUARES:
        rank, file = chess.square_rank(sq), chess.square_file(sq)
        pos_encoding = [file / 7.0, rank / 7.0]
        is_attacked_by_white = float(board.is_attacked_by(chess.WHITE, sq))
        is_attacked_by_black = float(board.is_attacked_by(chess.BLACK, sq))
        control_status = [is_attacked_by_white, is_attacked_by_black]

        local_features = np.concatenate([pos_encoding, control_status])
        square_features_list.append(np.concatenate([local_features, global_features]))

    data['square'].x = torch.from_numpy(np.array(square_features_list, dtype=np.float32))

    piece_map = board.piece_map()
    if not piece_map:
        data['piece'].x = torch.empty((0, PIECE_FEATURE_DIM))
    else:
        sorted_squares = sorted(piece_map.keys())
        piece_node_indices = {sq: i for i, sq in enumerate(sorted_squares)}

        piece_features_list = []
        piece_mobilities = {sq: 0 for sq in sorted_squares}
        if board.is_valid():
            try:
                for move in board.legal_moves:
                    if move.from_square in piece_mobilities:
                        piece_mobilities[move.from_square] += 1
            except (AssertionError, ValueError):
                pass

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
            material_value = [PIECE_MATERIAL_VALUE[piece.piece_type] / 9.0]

            local_features = np.concatenate([
                piece_type_one_hot, piece_color, location, mobility,
                attack_defense, material_value
            ])
            piece_features_list.append(np.concatenate([local_features, global_features]))

        data['piece'].x = torch.from_numpy(np.array(piece_features_list, dtype=np.float32))

    # --- 5. Edge Indices (for GNN) ---
    data['square', 'adjacent_to', 'square'].edge_index = _SQUARE_ADJACENCY_EDGE_INDEX.clone()

    occupies_edges, attacks_piece_edges, defends_piece_edges = [], [], []
    if piece_map:
        sorted_squares = sorted(piece_map.keys())
        piece_node_indices = {sq: i for i, sq in enumerate(sorted_squares)}
        for from_idx, from_sq in enumerate(sorted_squares):
            piece = board.piece_at(from_sq)
            occupies_edges.append([from_idx, from_sq])
            for to_sq in board.attacks(from_sq):
                target_piece = board.piece_at(to_sq)
                if target_piece and to_sq in piece_node_indices:
                    to_idx = piece_node_indices[to_sq]
                    if target_piece.color == piece.color:
                        defends_piece_edges.append([from_idx, to_idx])
                    else:
                        attacks_piece_edges.append([from_idx, to_idx])

    piece_self_loops = [[i, i] for i in range(len(piece_map))]

    data['piece', 'occupies', 'square'].edge_index = torch.tensor(occupies_edges, dtype=torch.long).t().contiguous() if occupies_edges else torch.empty((2, 0), dtype=torch.long)
    data['piece', 'attacks', 'piece'].edge_index = torch.tensor(attacks_piece_edges + piece_self_loops, dtype=torch.long).t().contiguous() if attacks_piece_edges or piece_self_loops else torch.empty((2, 0), dtype=torch.long)
    data['piece', 'defends', 'piece'].edge_index = torch.tensor(defends_piece_edges, dtype=torch.long).t().contiguous() if defends_piece_edges else torch.empty((2, 0), dtype=torch.long)

    # --- 6. Finalize and move to device ---
    return data.to(device), cnn_tensor.to(device), material_balance_tensor.to(device)