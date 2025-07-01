import pytest
import chess
import numpy as np
import torch

# Updated imports to include new batching function and match project structure
from gnn_agent.gamestate_converters.gnn_data_converter import (
    convert_to_gnn_input,
    convert_boards_to_gnn_batch,
    PIECE_TYPE_MAP,
    GNN_INPUT_FEATURE_DIM
)

def test_conversion_on_starting_position():
    """Tests the GNN data conversion on the standard chess starting position."""
    board = chess.Board()
    device = torch.device("cpu")
    gnn_input = convert_to_gnn_input(board, device)

    # --- 1. Validate Square Graph ---
    sq_graph = gnn_input.square_graph
    
    # Check node feature shape: 64 squares, 12 features each
    assert sq_graph.x.shape == (64, GNN_INPUT_FEATURE_DIM)
    
    # Check edge index shape (static adjacency)
    assert sq_graph.edge_index.shape[0] == 2
    assert sq_graph.edge_index.shape[1] > 0

    # Verify features for a specific square, e.g., E2 (square 12)
    e2_features = sq_graph.x[chess.E2]
    assert np.allclose(e2_features[0:2].numpy(), [4/7.0, 1/7.0])
    assert e2_features[2 + PIECE_TYPE_MAP[chess.PAWN]] == 1.0
    assert torch.sum(e2_features[2:8]) == 1.0
    assert e2_features[8] == 1.0 # White
    
    # Verify features for an empty square, e.g., E4 (square 28)
    e4_features = sq_graph.x[chess.E4]
    assert torch.sum(e4_features[2:8]) == 0.0
    assert torch.sum(e4_features[8:10]) == 0.0
    assert e4_features[10] == 0.0 # is_attacked_by_white
    assert e4_features[11] == 0.0 # is_attacked_by_black

    # --- 2. Validate Piece Graph ---
    pc_graph = gnn_input.piece_graph
    assert pc_graph.x.shape[0] == 32 # 32 pieces
    assert pc_graph.x.shape[1] == GNN_INPUT_FEATURE_DIM
    assert pc_graph.edge_index.shape[1] > 0 # Should have defensive edges

def test_conversion_on_mid_game_position():
    """Tests the GNN data conversion on a more complex mid-game position."""
    fen = "r1bqkbnr/1pp2ppp/p1np4/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5"
    board = chess.Board(fen)
    device = torch.device("cpu")
    gnn_input = convert_to_gnn_input(board, device)

    pc_graph = gnn_input.piece_graph
    num_pieces = len(board.piece_map())
    assert pc_graph.x.shape[0] == num_pieces
    assert pc_graph.edge_index.shape[1] > 0

    # Verify mobility for a specific piece, e.g., the white knight on f3
    knight_sq = chess.F3
    
    # Recreate node ordering from application logic to find correct index
    sorted_squares = sorted(board.piece_map().keys())
    piece_node_indices = {sq: i for i, sq in enumerate(sorted_squares)}
    knight_node_idx = piece_node_indices[knight_sq]
    
    knight_features = pc_graph.x[knight_node_idx]
    
    # Mobility feature is at index 9: [type(6), color(1), loc(2), mobility(1), atk/def(2)]
    mobility_feature = knight_features[9]
    # Knight on f3 has 5 legal moves: Ng5, Nh4, Ng1, Ne1, Nd2
    expected_mobility = 5 / 28.0
    assert np.isclose(mobility_feature.item(), expected_mobility)

# ===============================================================
# == NEW TESTS FOR BATCHING FUNCTIONALITY
# ===============================================================

def test_batch_converter_shapes_and_types():
    """
    Tests if the batched converter produces tensors with the correct shapes and dtypes for a standard batch.
    """
    board1 = chess.Board()
    board2 = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2") # After 1. e4 c5
    boards = [board1, board2]
    device = torch.device("cpu")
    
    batch = convert_boards_to_gnn_batch(boards, device)

    batch_size = len(boards)
    num_squares = 64
    num_pieces_b1 = len(board1.piece_map())
    num_pieces_b2 = len(board2.piece_map())
    total_pieces = num_pieces_b1 + num_pieces_b2

    # Check square features and batch vector
    assert batch.square_features.shape == (batch_size * num_squares, GNN_INPUT_FEATURE_DIM)
    assert batch.square_batch.shape == (batch_size * num_squares,)
    assert torch.all(batch.square_batch[:num_squares] == 0).item()
    assert torch.all(batch.square_batch[num_squares:] == 1).item()

    # Check piece features and batch vector
    assert batch.piece_features.shape == (total_pieces, GNN_INPUT_FEATURE_DIM)
    assert batch.piece_batch.shape == (total_pieces,)
    assert torch.all(batch.piece_batch[:num_pieces_b1] == 0).item()
    assert torch.all(batch.piece_batch[num_pieces_b1:] == 1).item()

    # Check padding mask
    max_pieces = max(num_pieces_b1, num_pieces_b2)
    assert batch.piece_padding_mask.shape == (batch_size, max_pieces)
    assert batch.piece_padding_mask.dtype == torch.bool
    assert not batch.piece_padding_mask[0, :num_pieces_b1].any() # Not masked
    assert not batch.piece_padding_mask[1, :num_pieces_b2].any() # Not masked
    
    # Check that piece_to_square_map is correctly offset
    assert batch.piece_to_square_map.max() < batch_size * num_squares

def test_batch_converter_with_empty_board():
    """
    Tests if the batched converter handles edge cases like an empty board in the batch.
    """
    board1 = chess.Board()
    board2 = chess.Board(fen=None) # Empty board
    boards = [board1, board2]
    device = torch.device("cpu")

    batch = convert_boards_to_gnn_batch(boards, device)
    
    batch_size = len(boards)
    num_pieces_b1 = len(board1.piece_map())
    
    # The total number of pieces in the batch should only be from board1
    assert batch.piece_features.shape[0] == num_pieces_b1
    assert torch.all(batch.piece_batch == 0).item()
    
    # Padding mask should correctly mask all pieces for the second item in the batch
    max_pieces = num_pieces_b1
    assert batch.piece_padding_mask.shape == (batch_size, max_pieces)
    assert batch.piece_padding_mask[1, :].all() # All masked for the empty board