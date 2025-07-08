import unittest
from unittest.mock import MagicMock
import chess
import torch
from gnn_agent.rl_loop.guided_session import _decide_agent_action

class TestDecideAgentAction(unittest.TestCase):

    def setUp(self):
        """Set up mock objects for each test."""
        self.board = chess.Board()
        # Mock Agent and its components
        self.mock_agent = MagicMock()
        self.mock_network = MagicMock()
        self.mock_agent.network = self.mock_network
        self.mock_converter = MagicMock()
        mock_gnn_data = MagicMock()
        mock_gnn_data.to.return_value = mock_gnn_data
        self.mock_converter.convert.return_value = (mock_gnn_data, None)
        self.mock_agent.gnn_data_converter = self.mock_converter
        # Mock Mentor Engine
        self.mock_mentor = MagicMock()
        self.mock_mentor.get_evaluation.return_value = {'type': 'cp', 'value': 150} # tanh(0.15) ~= 0.148
        # Mock MCTS Search Manager
        self.mock_search_manager = MagicMock()
        self.policy_tensor = torch.zeros(4672)
        self.policy_tensor[0] = 1.0
        self.mock_search_manager.run_search.return_value = (self.policy_tensor, 0.5)

    def test_intervention_triggers_when_moves_differ(self):
        """
        If agent and mentor moves differ, the mentor's move should be chosen.
        """
        self.mock_agent.action_to_move.return_value = chess.Move.from_uci('e2e4')
        self.mock_mentor.get_best_move_time.return_value = 'g1f3' # Different move
        # Agent's value is high, so discrepancy will be high, and guided mode should continue
        self.mock_network.return_value = (None, torch.tensor([[0.8]]))

        move, still_guided, policy = _decide_agent_action(
            self.mock_agent, self.mock_mentor, self.board, self.mock_search_manager,
            in_guided_mode=True, value_threshold=0.1
        )

        self.assertEqual(move, chess.Move.from_uci('g1f3')) # Assert mentor's move was chosen
        self.assertTrue(still_guided) # Assert guided mode continues
        self.assertIsNotNone(policy) # Assert policy is returned for the buffer

    def test_guided_mode_ends_when_moves_align(self):
        """
        If agent and mentor moves are the same, guided mode should end.
        """
        self.mock_agent.action_to_move.return_value = chess.Move.from_uci('e2e4')
        self.mock_mentor.get_best_move_time.return_value = 'e2e4' # Same move

        move, still_guided, _ = _decide_agent_action(
            self.mock_agent, self.mock_mentor, self.board, self.mock_search_manager,
            in_guided_mode=True, value_threshold=0.1
        )

        self.assertEqual(move, chess.Move.from_uci('e2e4')) # Assert agent's move was chosen
        self.assertFalse(still_guided) # Assert guided mode ENDS

    def test_guided_mode_ends_when_value_aligns(self):
        """
        After an intervention, if value discrepancy is low, guided mode should end.
        """
        self.mock_agent.action_to_move.return_value = chess.Move.from_uci('e2e4')
        self.mock_mentor.get_best_move_time.return_value = 'g1f3' # Different move
        # Agent's value (0.1) is close to mentor's (0.148), so discrepancy is low
        self.mock_network.return_value = (None, torch.tensor([[0.1]]))

        move, still_guided, _ = _decide_agent_action(
            self.mock_agent, self.mock_mentor, self.board, self.mock_search_manager,
            in_guided_mode=True, value_threshold=0.2 # Higher threshold to ensure pass
        )

        self.assertEqual(move, chess.Move.from_uci('g1f3')) # Assert mentor's move was chosen
        self.assertFalse(still_guided) # Assert guided mode ENDS because value aligned

    def test_no_guidance_when_not_in_guided_mode(self):
        """
        If not in guided mode, the agent's move is chosen and mentor is not called.
        """
        self.mock_agent.action_to_move.return_value = chess.Move.from_uci('e2e4')

        move, still_guided, _ = _decide_agent_action(
            self.mock_agent, self.mock_mentor, self.board, self.mock_search_manager,
            in_guided_mode=False, value_threshold=0.1 # Should be ignored
        )

        self.assertEqual(move, chess.Move.from_uci('e2e4'))
        self.assertFalse(still_guided)
        self.mock_mentor.get_best_move_time.assert_not_called() # Crucial assertion

if __name__ == '__main__':
    unittest.main()