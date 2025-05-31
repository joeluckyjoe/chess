# test_gnn_data_converter.py

import pytest
import chess
import numpy as np
from gnn_data_converter import convert_to_gnn_input, PIECE_TYPE_MAP

def test_conversion_on_starting_position():
    """Tests the GNN data conversion on the standard chess starting position."""
    board = chess.Board()
    gnn_input = convert_to_gnn_input(board)

    # --- 1. Validate Square Graph ---
    sq_graph = gnn_input.square_graph
    
    # Check node feature shape: 64 squares, 12 features each
    assert sq_graph.node_features.shape == (64, 12)
    
    # Check edge index shape (static adjacency)
    # The exact number of edges is fixed, let's check one connection
    # e.g., A1 (0) should be connected to B1 (1), A2 (8), B2 (9)
    assert sq_graph.edge_index.shape[0] == 2
    assert sq_graph.edge_index.shape[1] > 0 # Should have edges
    
    # Verify features for a specific square, e.g., E2 (square 12)
    e2_features = sq_graph.node_features[chess.E2]
    # Positional encoding for E2 (file 4, rank 1)
    assert np.allclose(e2_features[0:2], [4/7.0, 1/7.0])
    # Piece type: Pawn (index 0)
    assert e2_features[2 + PIECE_TYPE_MAP[chess.PAWN]] == 1.0
    assert np.sum(e2_features[2:8]) == 1.0 # Only one piece type bit is set
    # Piece color: White (index 0)
    assert e2_features[8] == 1.0
    assert e2_features[9] == 0.0
    # Attack status: E2 is attacked by D1 (Queen) and F1 (King) for white
    # (for castling rights logic) but not black.
    assert e2_features[10] == 1.0 # is_attacked_by_white
    assert e2_features[11] == 0.0 # is_attacked_by_black

    # Verify features for an empty square, e.g., E4 (square 28)
    e4_features = sq_graph.node_features[chess.E4]
    # No piece type or color
    assert np.sum(e4_features[2:8]) == 0.0
    assert np.sum(e4_features[8:10]) == 0.0
    # In the starting position, E4 is NOT attacked by white or black.
    assert e4_features[10] == 0.0 # is_attacked_by_white
    assert e4_features[11] == 0.0 # is_attacked_by_black

    # --- 2. Validate Piece Graph ---
    pc_graph = gnn_input.piece_graph

    # In the starting position, there are 32 pieces
    assert pc_graph.node_features.shape[0] == 32
    # 12 features per piece
    assert pc_graph.node_features.shape[1] == 12
    
    # In starting position, no piece attacks another, so edge_index should be empty
    # In the starting position, all edges should be defensive (same-colored pieces).
    # There should be no attacks on opponents.
    assert pc_graph.edge_index.shape[1] > 0 # Make sure we have edges

    # Get an ordered list of pieces corresponding to the node indices
    pieces_in_order = list(board.piece_map().values())

    # Check every edge
    for i in range(pc_graph.edge_index.shape[1]):
        source_node_idx = pc_graph.edge_index[0, i]
        dest_node_idx = pc_graph.edge_index[1, i]

        source_piece = pieces_in_order[source_node_idx]
        dest_piece = pieces_in_order[dest_node_idx]

        # Assert that the source and destination pieces have the same color
        assert source_piece.color == dest_piece.color
    
def test_conversion_on_mid_game_position():
    """Tests the GNN data conversion on a more complex mid-game position."""
    # Ruy-Lopez opening after a few moves
    fen = "r1bqkbnr/1pp2ppp/p1np4/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5"
    board = chess.Board(fen)
    gnn_input = convert_to_gnn_input(board)

    # --- 1. Validate Square Graph ---
    sq_graph = gnn_input.square_graph
    assert sq_graph.node_features.shape == (64, 12)

    # --- 2. Validate Piece Graph ---
    pc_graph = gnn_input.piece_graph
    
    # Count pieces on the board
    num_pieces = len(board.piece_map())
    assert pc_graph.node_features.shape[0] == num_pieces
    assert pc_graph.node_features.shape[1] == 12

    # Check that there are now edges
    # e.g., white Knight on f3 attacks black pawn on e5
    assert pc_graph.edge_index.shape[0] == 2
    assert pc_graph.edge_index.shape[1] > 0

    # Verify mobility for a specific piece, e.g., the white knight on f3
    knight_sq = chess.F3
    piece_map = board.piece_map()
    piece_node_indices = {sq: i for i, sq in enumerate(piece_map.keys())}
    knight_node_idx = piece_node_indices[knight_sq]
    
    knight_features = pc_graph.node_features[knight_node_idx]
    
    # Find mobility feature (index 9: 6 for type + 1 for color + 2 for loc)
    mobility_feature = knight_features[9]
    # Knight on f3 has 5 legal moves: Ng5, Nh4, Ng1, Ne1, Nd2
    expected_mobility = 5 / 28.0
    assert np.isclose(mobility_feature, expected_mobility)