#
# File: gnn_agent/gamestate_converters/gnn_data_converter.py (Updated for Phase AV)
#
import torch
import chess
import numpy as np
from typing import Dict
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# --- Constants for Feature Engineering ---
# --- PHASE AV MODIFICATION: Add 2 global features (Material Balance, Repetition Counter) ---
SQUARE_FEATURE_DIM = 21  # Was 19
PIECE_FEATURE_DIM = 15   # Was 13
# --- END MODIFICATION ---

PIECE_TYPE_MAP: Dict[chess.PieceType, int] = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}
NUM_PIECE_TYPES = len(PIECE_TYPE_MAP)

PIECE_MATERIAL_VALUE: Dict[chess.PieceType, float] = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,  # King has no material value
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

# --- Main Conversion Function ---

def convert_to_gnn_input(board: chess.Board, device: torch.device) -> Data:
    """
    Converts a single python-chess board state into a PyTorch Geometric Data object.
    This version uses two distinct, dense feature sets for squares and pieces.
    """
    # --- Original Global Game State Features ---
    turn = 1.0 if board.turn == chess.WHITE else 0.0
    can_castle_wk = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    can_castle_wq = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    can_castle_bk = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    can_castle_bq = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    ep_square = board.ep_square
    en_passant_file = (chess.square_file(ep_square) / 7.0) if ep_square is not None else 0.0
    halfmove_clock = board.halfmove_clock / 100.0
    
    original_global_features = np.array([
        turn, can_castle_wk, can_castle_wq, can_castle_bk, can_castle_bq,
        en_passant_file, halfmove_clock
    ], dtype=np.float32)

    # --- PHASE AV MODIFICATION: Calculate Material Balance and Repetition Count ---
    white_material = 0.0
    black_material = 0.0
    for piece in board.piece_map().values():
        value = PIECE_MATERIAL_VALUE[piece.piece_type]
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value
    
    # Normalize by the total starting material (39 for each side)
    # The value is from the current player's perspective
    raw_balance = white_material - black_material
    perspective_balance = raw_balance if board.turn == chess.WHITE else -raw_balance
    normalized_balance = perspective_balance / 39.0

    # Check if a draw by threefold repetition can be claimed
    repetition_counter = 1.0 if board.can_claim_threefold_repetition() else 0.0
    
    new_global_features = np.array([normalized_balance, repetition_counter], dtype=np.float32)

    # Combine all global features
    all_global_features = np.concatenate([original_global_features, new_global_features])
    # --- END MODIFICATION ---

    # 1. Square-based Graph Features (G_sq)
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
        
        local_features = np.concatenate([pos_encoding, piece_type_one_hot, piece_color_one_hot, control_status])
        # --- PHASE AV MODIFICATION: Use the combined global features ---
        square_features_list.append(np.concatenate([local_features, all_global_features]))
        # --- END MODIFICATION ---

    square_features = torch.from_numpy(np.array(square_features_list, dtype=np.float32))
    square_edge_index = _SQUARE_ADJACENCY_EDGE_INDEX.clone()

    # 2. Piece-based Graph Features (G_pc)
    piece_map = board.piece_map()
    num_pieces = len(piece_map)
    
    if num_pieces == 0:
        piece_features = torch.empty((0, PIECE_FEATURE_DIM))
        piece_edge_index = torch.empty((2, 0), dtype=torch.long)
        piece_to_square_map = torch.empty((0), dtype=torch.long)
    else:
        sorted_squares = sorted(piece_map.keys())
        piece_node_indices = {sq: i for i, sq in enumerate(sorted_squares)}
        piece_to_square_map = torch.tensor(sorted_squares, dtype=torch.long)
        
        piece_features_list, piece_edges = [], []
        piece_mobilities = {sq: 0 for sq in sorted_squares}
        if board.is_valid():
            try:
                for move in board.legal_moves:
                    if move.from_square in piece_mobilities:
                        piece_mobilities[move.from_square] += 1
            except (AssertionError, ValueError):
                # In rare cases with invalid FENs, legal_moves can fail.
                # We can proceed with mobilities as 0.
                pass
        
        for from_sq in sorted_squares:
            piece = board.piece_at(from_sq)
            piece_type_one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
            piece_type_one_hot[PIECE_TYPE_MAP[piece.piece_type]] = 1.0
            piece_color = [1.0 if piece.color == chess.WHITE else 0.0]
            rank, file = chess.square_rank(from_sq), chess.square_file(from_sq)
            location = [file / 7.0, rank / 7.0]
            mobility = [piece_mobilities.get(from_sq, 0) / 28.0]  # Normalize by max possible moves
            attack_count = len(board.attacks(from_sq) & board.occupied_co[not piece.color])
            defense_count = len(board.attackers(piece.color, from_sq))
            attack_defense = [float(attack_count), float(defense_count)]
            material_value = [PIECE_MATERIAL_VALUE[piece.piece_type] / 9.0]  # Normalize by queen value
            
            local_features = np.concatenate([
                piece_type_one_hot, piece_color, location, mobility,
                attack_defense, material_value
            ])
            
            # --- PHASE AV MODIFICATION: Add global features to each piece node ---
            # Per the plan, both graphs get the new features to provide full context.
            full_piece_features = np.concatenate([local_features, new_global_features])
            piece_features_list.append(full_piece_features)
            # --- END MODIFICATION ---
            
            for to_sq in board.attacks(from_sq):
                if to_sq in piece_node_indices:
                    piece_edges.append((piece_node_indices[from_sq], piece_node_indices[to_sq]))
            
        piece_features = torch.from_numpy(np.array(piece_features_list, dtype=np.float32))
        piece_edge_index_no_loops = torch.tensor(piece_edges, dtype=torch.long).t().contiguous() if piece_edges else torch.empty((2, 0), dtype=torch.long)
        
        piece_edge_index, _ = add_self_loops(piece_edge_index_no_loops, num_nodes=num_pieces)

    # 3. Create the PyG Data object
    data = Data(
        square_features=square_features,
        square_edge_index=square_edge_index,
        piece_features=piece_features,
        piece_edge_index=piece_edge_index,
        piece_to_square_map=piece_to_square_map,
        num_nodes=num_pieces 
    )
    
    return data.to(device)