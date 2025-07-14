import pytest
import torch
import chess
from torch_geometric.data import HeteroData

from gnn_agent.gamestate_converters.gnn_data_converter import (
    convert_to_gnn_input,
    CNN_INPUT_CHANNELS,
    SQUARE_FEATURE_DIM,
    PIECE_FEATURE_DIM
)

@pytest.fixture
def starting_board():
    """Provides a standard chess board in its starting position."""
    return chess.Board()

@pytest.fixture
def empty_board():
    """Provides an empty chess board."""
    return chess.Board(fen=None)

@pytest.fixture
def device():
    """Provides a torch device (CPU for testing)."""
    return torch.device("cpu")


def test_converter_return_type(starting_board, device):
    """
    Tests that the converter returns a tuple of (HeteroData, torch.Tensor).
    """
    result = convert_to_gnn_input(starting_board, device)
    assert isinstance(result, tuple), "Output should be a tuple."
    assert len(result) == 2, "Output tuple should have two elements."
    assert isinstance(result[0], HeteroData), "First element should be a HeteroData object."
    assert isinstance(result[1], torch.Tensor), "Second element should be a torch.Tensor."

def test_cnn_tensor_shape_and_type(starting_board, device):
    """
    Tests that the CNN tensor has the correct shape, dtype, and device.
    """
    _, cnn_tensor = convert_to_gnn_input(starting_board, device)
    expected_shape = (CNN_INPUT_CHANNELS, 8, 8)
    assert cnn_tensor.shape == expected_shape, \
        f"CNN tensor shape is incorrect. Expected {expected_shape}, got {cnn_tensor.shape}."
    assert cnn_tensor.dtype == torch.float32, "CNN tensor dtype should be float32."
    assert str(cnn_tensor.device) == str(device), "CNN tensor is on the wrong device."

def test_gnn_data_structure(starting_board, device):
    """
    Tests the basic structure and feature dimensions of the HeteroData object.
    """
    gnn_data, _ = convert_to_gnn_input(starting_board, device)
    
    # Check node features
    assert 'square' in gnn_data.node_types
    assert 'piece' in gnn_data.node_types
    assert gnn_data['square'].x.shape == (64, SQUARE_FEATURE_DIM)
    assert gnn_data['piece'].x.shape == (32, PIECE_FEATURE_DIM) # 32 pieces at start
    assert gnn_data['square'].x.dtype == torch.float32
    assert gnn_data['piece'].x.dtype == torch.float32

    # Check edge types
    assert ('piece', 'attacks', 'piece') in gnn_data.edge_types
    assert ('piece', 'occupies', 'square') in gnn_data.edge_types
    
    # Check device
    assert str(gnn_data['square'].x.device) == str(device)
    assert str(gnn_data['piece', 'attacks', 'piece'].edge_index.device) == str(device)

def test_cnn_tensor_content_starting_board(starting_board, device):
    """
    Sanity-checks the content of the CNN tensor for the starting position.
    """
    _, cnn_tensor = convert_to_gnn_input(starting_board, device)

    # White's turn, so channel 12 should be all 1s
    assert torch.all(cnn_tensor[12, :, :] == 1.0), "Turn channel incorrect for White's move."

    # halfmove_clock is 0 at start, so channel 13 should be all 0s
    assert torch.all(cnn_tensor[13, :, :] == 0.0), "Half-move clock channel incorrect for start."
    
    # Check for a white pawn on e2 (rank 1, file 4)
    # Channel 0 corresponds to white pawns
    assert cnn_tensor[0, 1, 4] == 1.0, "White pawn not found on e2 in CNN tensor."
    
    # Check for a black knight on g8 (rank 7, file 6)
    # Channel 7 (1 + 6) corresponds to black knights
    assert cnn_tensor[7, 7, 6] == 1.0, "Black knight not found on g8 in CNN tensor."

def test_converter_on_empty_board(empty_board, device):
    """
    Tests that the converter runs without error on a board with no pieces.
    """
    try:
        gnn_data, cnn_tensor = convert_to_gnn_input(empty_board, device)
        assert gnn_data['piece'].x.shape[0] == 0
        assert gnn_data['piece', 'attacks', 'piece'].edge_index.shape[1] == 0
        # The first 12 channels (pieces) should be all zeros
        assert torch.sum(cnn_tensor[:12, :, :]) == 0
    except Exception as e:
        pytest.fail(f"Converter failed on empty board: {e}")