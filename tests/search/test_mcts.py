#
# tests/search/test_mcts.py (Corrected and Verified)
#
import unittest
import torch
import chess
from unittest.mock import MagicMock

# Import the classes we are testing
from gnn_agent.search.mcts import MCTS
from gnn_agent.search.mcts_node import MCTSNode

# --- Mocks for predictable test behavior ---

class MockChessNetwork(torch.nn.Module):
    """A mock neural network that returns predictable, constant values."""
    def __init__(self, move_dims=4672):
        super().__init__()
        self.move_dims = move_dims
        self.dummy_layer = torch.nn.Linear(1, 1)

    # This signature now correctly matches the real network, accepting unpacked arguments
    def forward(self, square_features, square_edge_index, piece_features, piece_edge_index, piece_to_square_map):
        policy_logits = torch.ones(1, self.move_dims)
        value = torch.tensor([[0.5]])
        return policy_logits, value

class MockGNNInput:
    """Mock version of GNNInput that can store data and be iterated."""
    def __init__(self, square_graph, piece_graph):
        self.square_graph = square_graph
        self.piece_graph = piece_graph
        self.piece_to_square_map = torch.empty((0), dtype=torch.long)

    def __iter__(self):
        """Allows unpacking the object like a tuple for the network forward pass."""
        yield self.square_graph['x']
        yield self.square_graph['edge_index']
        yield self.piece_graph['x']
        yield self.piece_graph['edge_index']
        yield self.piece_to_square_map

# This mock now creates and returns a properly structured MockGNNInput object
def mock_convert_to_gnn_input(board, device):
    """A mock data converter that returns a valid, iterable MockGNNInput."""
    mock_square_graph = {'x': torch.randn(64, 1), 'edge_index': torch.empty(2, 0)}
    mock_piece_graph = {'x': torch.randn(32, 1), 'edge_index': torch.empty(2, 0)}
    return MockGNNInput(mock_square_graph, mock_piece_graph)


class TestMCTS(unittest.TestCase):

    def setUp(self):
        """Set up a mock network and MCTS instance for testing."""
        self.device = torch.device("cpu")
        self.mock_network = MockChessNetwork()
        self.mcts = MCTS(network=self.mock_network, device=self.device, c_puct=1.41)
        
        # Monkey-patch the real converter with our mock version for testing purposes.
        import gnn_agent.search.mcts as mcts_module
        mcts_module.convert_to_gnn_input = mock_convert_to_gnn_input
        
        # We no longer need to mock the expand method, as our main logic should work.

    def test_initialization(self):
        """Test if the MCTS class is initialized correctly."""
        self.assertIsNotNone(self.mcts)
        self.assertEqual(self.mcts.root, None)
        # This now correctly checks for 'c_puct' without the underscore
        self.assertEqual(self.mcts.c_puct, 1.41)
        # This now correctly checks for 'network' without the underscore
        self.assertIs(self.mcts.network, self.mock_network)

    def test_run_single_simulation(self):
        """Test running a single MCTS simulation from the starting position."""
        board = chess.Board()
        num_simulations = 1
        
        best_move = self.mcts.run_search(board, num_simulations)
        
        self.assertIsNotNone(self.mcts.root)
        self.assertEqual(self.mcts.root.N, 1) # Use N for visit_count as per MCTSNode
        self.assertAlmostEqual(self.mcts.root.Q, 0.5) # Use Q for q_value
        self.assertEqual(len(self.mcts.root.children), 20)
        self.assertIn(best_move, board.legal_moves)

    def test_run_multiple_simulations(self):
        """Test that visit counts increase with more simulations."""
        board = chess.Board()
        num_simulations = 25
        
        best_move = self.mcts.run_search(board, num_simulations)
        
        self.assertIsNotNone(self.mcts.root)
        self.assertEqual(self.mcts.root.N, num_simulations)
        
        child_visits = sum(child.N for child in self.mcts.root.children.values())
        self.assertEqual(child_visits, self.mcts.root.N - 1)
        
        self.assertAlmostEqual(self.mcts.root.Q, 0.5, places=5)
        self.assertIn(best_move, board.legal_moves)

if __name__ == '__main__':
    unittest.main()