import torch
import chess
import pytest
from unittest.mock import MagicMock

from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.gamestate_converters.gnn_data_converter import convert_boards_to_gnn_batch
from gnn_agent.gamestate_converters.action_space_converter import ACTION_SPACE_SIZE

def test_chess_network_batch_forward_pass():
    """
    Tests that the ChessNetwork can process a batch without runtime errors
    and that the output tensors have the correct batch dimension.
    """
    batch_size = 2
    device = torch.device("cpu")
    
    # --- 1. Create Mock Sub-Modules ---
    mock_square_gnn = MagicMock()
    mock_piece_gnn = MagicMock()
    mock_cross_attention = MagicMock()
    mock_policy_head = MagicMock()
    mock_value_head = MagicMock()

    # --- 2. Configure Mock Return Values ---
    sq_embedding_dim = 128
    pc_embedding_dim = 128
    
    # --- THIS IS THE FIX ---
    # We must explicitly set the dimension attributes on the mock attention module
    # so the ChessNetwork constructor can correctly size its linear layer.
    mock_cross_attention.sq_embed_dim = sq_embedding_dim
    mock_cross_attention.pc_embed_dim = pc_embedding_dim
    # --- END FIX ---
    
    mock_square_gnn.return_value = torch.randn(batch_size * 64, sq_embedding_dim)
    mock_piece_gnn.return_value = torch.randn(batch_size * 32, pc_embedding_dim)
    
    mock_cross_attention.return_value = (
        torch.randn(batch_size, 32, pc_embedding_dim),
        torch.randn(batch_size, 64, sq_embedding_dim),
        None, None
    )

    mock_policy_head.return_value = torch.randn(batch_size, ACTION_SPACE_SIZE)
    mock_value_head.return_value = torch.randn(batch_size, 1)

    # --- 3. Instantiate the Real Network with Mocks ---
    network = ChessNetwork(
        square_gnn=mock_square_gnn,
        piece_gnn=mock_piece_gnn,
        cross_attention=mock_cross_attention,
        policy_head=mock_policy_head,
        value_head=mock_value_head
    ).to(device)

    # --- 4. Create a Realistic Batch Input ---
    boards = [chess.Board(), chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")]
    batch = convert_boards_to_gnn_batch(boards, device)

    # --- 5. Run Forward Pass ---
    policy_logits, value = network(
        square_features=batch.square_features,
        square_edge_index=batch.square_edge_index,
        square_batch=batch.square_batch,
        piece_features=batch.piece_features,
        piece_edge_index=batch.piece_edge_index,
        piece_batch=batch.piece_batch,
        piece_to_square_map=batch.piece_to_square_map,
        piece_padding_mask=batch.piece_padding_mask
    )

    # --- 6. Assert Output Shapes ---
    assert policy_logits.shape == (batch_size, ACTION_SPACE_SIZE)
    assert value.shape == (batch_size, 1)