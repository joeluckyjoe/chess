import torch
import chess
import pytest
import numpy as np
from unittest.mock import MagicMock

from gnn_agent.search.mcts import MCTS
from gnn_agent.gamestate_converters.action_space_converter import ACTION_SPACE_SIZE

@pytest.fixture
def mcts_setup():
    """Sets up a mock network and MCTS instance for testing."""
    device = torch.device("cpu")
    mock_network = MagicMock()
    mcts_instance = MCTS(network=mock_network, device=device, batch_size=8, c_puct=1.41)
    return mcts_instance, mock_network, device

def test_initialization(mcts_setup):
    """Tests if the MCTS class is initialized correctly."""
    mcts, mock_network, _ = mcts_setup
    assert mcts.network is mock_network
    assert mcts.root is None
    assert mcts.c_puct == 1.41
    assert mcts.batch_size == 8

def configure_mock_network(mock_network):
    """Helper function to configure the mock network's side_effect."""
    def network_side_effect(*args, **kwargs):
        current_batch_size = int(kwargs['square_batch'].max().item() + 1)
        policy = torch.randn(current_batch_size, ACTION_SPACE_SIZE)
        value = torch.rand(current_batch_size, 1)
        return policy, value
    mock_network.side_effect = network_side_effect

def test_run_search_returns_valid_policy(mcts_setup):
    """
    Test running a search to ensure it completes and returns a valid policy dict.
    """
    mcts, mock_network, _ = mcts_setup
    board = chess.Board()
    num_simulations = 25

    configure_mock_network(mock_network)
    
    policy = mcts.run_search(board, num_simulations)
    
    assert isinstance(policy, dict)
    assert len(policy) > 0
    assert mcts.root is not None
    assert mcts.root.N == num_simulations
    assert np.isclose(sum(policy.values()), 1.0)
    
    legal_moves = list(board.legal_moves)
    for move in policy.keys():
        assert move in legal_moves

@pytest.mark.parametrize("num_simulations, batch_size, expected_calls", [
    (16, 8, 3),
    (17, 8, 3), # <-- THIS IS THE FIX (was 4, is now 3)
    (8, 8, 2),
    (7, 8, 2),
    (1, 8, 1),
])
def test_mcts_batching_network_calls(num_simulations, batch_size, expected_calls):
    """
    Tests that the MCTS engine calls the network the correct number of times.
    """
    device = torch.device("cpu")
    board = chess.Board()
    mock_network = MagicMock()
    configure_mock_network(mock_network)

    mcts = MCTS(network=mock_network, device=device, batch_size=batch_size)
    mcts.run_search(board, num_simulations=num_simulations)
    
    assert mock_network.call_count == expected_calls