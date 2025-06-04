import unittest
from unittest.mock import patch, MagicMock
import chess
import numpy as np
import torch

# CORRECTED: Import MCTS from its actual location within the gnn_agent package
from gnn_agent.search.mcts import MCTS

# ---- Fake MCTSNode for Testing ----
# This is a simplified version of your MCTSNode class, created just for this test.
# It mimics the interface that the MCTS class expects.
class FakeMCTSNode:
    def __init__(self, parent, prior_p, board_turn_at_node):
        self.parent = parent
        self.prior_p = prior_p
        self.board_turn_at_node = board_turn_at_node
        self.children = {}
        self.priors = {}
        self.N = 0
        self.Q = 0.0

    def is_leaf(self):
        return not self.children

    def expand(self, legal_moves, policy_priors, child_turn):
        self.priors = policy_priors
        for move, prior in policy_priors.items():
            # When expanding, create new FakeMCTSNode instances for children
            self.children[move] = FakeMCTSNode(parent=self, prior_p=prior, board_turn_at_node=child_turn)
    
    def uct_value(self, c_puct):
        # A simple UCT implementation for the test to run without errors.
        if self.N == 0:
            return float('inf')
        q_value = -self.Q / self.N
        exploration = c_puct * self.prior_p * (np.sqrt(self.parent.N) / (1 + self.N))
        return q_value + exploration

# ---- The Actual Unit Test ----
# CORRECTED: The patch paths now point to where the objects are looked up, inside gnn_agent.search.mcts
@patch('gnn_agent.search.mcts.MCTSNode', new=FakeMCTSNode)
class TestMCTSNoiseIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a mock network and a standard MCTS instance for testing."""
        self.board = chess.Board()
        self.device = torch.device('cpu')
        
        # Create a mock network that returns a fixed, uniform policy
        self.mock_network = MagicMock()
        
        legal_moves = list(self.board.legal_moves)
        self.num_legal_moves = len(legal_moves)
        
        # Mock the network's output logits
        # In a real scenario, these would be unnormalized. Softmax is applied in MCTS.
        # This size comes from your action_space_converter.py
        mock_logits = torch.ones(4672, device=self.device) 
        # The MCTS class expects a tuple of (policy_logits, value_tensor)
        self.mock_network.return_value = (mock_logits, torch.tensor(0.5)) 

        # CORRECTED: Patch the converter functions where they are used inside the mcts module
        self.mock_move_to_index = patch('gnn_agent.search.mcts.move_to_index').start()
        # Ensure moves map to unique indices for a clean policy
        self.mock_move_to_index.side_effect = lambda move, board: list(board.legal_moves).index(move)

        self.mock_convert_to_gnn = patch('gnn_agent.search.mcts.convert_to_gnn_input').start()
        self.mock_convert_to_gnn.return_value = (None, None, None) # Return dummy tuple

    def tearDown(self):
        """Stop all patches."""
        patch.stopall()

    def test_dirichlet_noise_is_applied_at_root(self):
        """
        Verify that Dirichlet noise is added to the root node's child priors
        after the initial expansion.
        """
        # Arrange
        dirichlet_alpha = 0.3
        dirichlet_epsilon = 0.25
        np.random.seed(42) # For reproducible noise

        # Instantiate MCTS with noise parameters
        mcts = MCTS(self.mock_network, self.device, dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon)
        
        # In your mcts.py, the network is used directly, not through predict method
        mcts.network = self.mock_network

        # Act
        # Run search for just one simulation. This is enough to trigger root expansion and noise application.
        mcts.run_search(self.board, num_simulations=1)

        # Assert
        self.assertIsNotNone(mcts.root, "Root node should have been created.")
        self.assertFalse(mcts.root.is_leaf(), "Root node should have been expanded.")
        
        # --- Calculate the expected priors ---
        # 1. Get the raw priors from the network after softmax. Since the input logits were uniform (all ones),
        # the softmax output will also be uniform.
        raw_uniform_prior = 1.0 / self.num_legal_moves
        
        # 2. Calculate the expected noise
        np.random.seed(42) # Reset seed to get the same noise
        expected_noise = np.random.dirichlet([dirichlet_alpha] * self.num_legal_moves)
        
        # 3. Calculate the final noisy priors
        expected_noisy_priors = (1 - dirichlet_epsilon) * raw_uniform_prior + dirichlet_epsilon * expected_noise

        # --- Get the actual priors from the MCTS run ---
        children = mcts.root.children.values()
        actual_noisy_priors = np.array([child.prior_p for child in children])

        # We must sort both actual and expected priors to ensure a correct comparison,
        # as the order of children in the dict is not guaranteed.
        actual_noisy_priors.sort()
        expected_noisy_priors.sort()
        
        # Final checks
        self.assertEqual(len(actual_noisy_priors), self.num_legal_moves)
        self.assertAlmostEqual(np.sum(actual_noisy_priors), 1.0, places=5, msg="Priors should still sum to 1.")
        
        # The core assertion: do the priors in the tree match what we expect after noise?
        np.testing.assert_allclose(actual_noisy_priors, expected_noisy_priors, rtol=1e-7,
                                   err_msg="The actual priors in the child nodes do not match the expected noisy priors.")


if __name__ == '__main__':
    unittest.main()