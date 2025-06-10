# tests/test_rl_loop/test_training_supervisor.py

import unittest
from gnn_agent.rl_loop.training_supervisor import TrainingSupervisor

class TestTrainingSupervisor(unittest.TestCase):
    """
    Unit tests for the TrainingSupervisor class.
    """

    def setUp(self):
        """
        Set up a sample configuration and instantiate the supervisor.
        """
        self.config = {
            'supervisor_loss_history_size': 5, # Use a small size for easy testing
            'bcp_threshold': 0.85
        }
        self.supervisor = TrainingSupervisor(self.config)

    def test_initialization(self):
        """
        Test that the supervisor initializes with the correct parameters from the config.
        """
        self.assertIsNotNone(self.supervisor)
        self.assertEqual(self.supervisor.loss_history_size, 5)
        self.assertEqual(self.supervisor.bcp_threshold, 0.85)
        self.assertEqual(len(self.supervisor.self_play_policy_losses), 0)

    def test_record_self_play_loss(self):
        """
        Test that losses are recorded correctly.
        """
        self.supervisor.record_self_play_loss(0.5)
        self.supervisor.record_self_play_loss(0.4)
        self.assertEqual(len(self.supervisor.self_play_policy_losses), 2)
        self.assertIn(0.5, self.supervisor.self_play_policy_losses)
        self.assertIn(0.4, self.supervisor.self_play_policy_losses)

    def test_loss_history_size_limit(self):
        """
        Test that the loss history deque respects the maximum size limit.
        """
        initial_losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for loss in initial_losses:
            self.supervisor.record_self_play_loss(loss)

        self.assertEqual(len(self.supervisor.self_play_policy_losses), 5)
        self.assertEqual(self.supervisor.self_play_policy_losses[0], 1.0) # Oldest element

        # Add one more loss, which should push out the oldest one (1.0)
        self.supervisor.record_self_play_loss(0.5)
        self.assertEqual(len(self.supervisor.self_play_policy_losses), 5)
        self.assertEqual(self.supervisor.self_play_policy_losses[0], 0.9) # The new oldest element
        self.assertNotIn(1.0, self.supervisor.self_play_policy_losses)
        self.assertIn(0.5, self.supervisor.self_play_policy_losses)

    def test_should_switch_to_mentor_default_behavior(self):
        """
        Test the placeholder behavior of should_switch_to_mentor. It should return False.
        """
        # Case 1: Not enough data
        self.supervisor.record_self_play_loss(0.5)
        self.assertFalse(self.supervisor.should_switch_to_mentor())

        # Case 2: Enough data, but logic is not implemented yet
        for i in range(5):
             self.supervisor.record_self_play_loss(0.5 - i * 0.01)
        self.assertFalse(self.supervisor.should_switch_to_mentor())

    def test_should_switch_to_self_play_default_behavior(self):
        """
        Test the placeholder behavior of should_switch_to_self_play. It should return False.
        """
        mentor_result = {'outcome': 0.5, 'game_length': 60}
        self.assertFalse(self.supervisor.should_switch_to_self_play(mentor_result))

if __name__ == '__main__':
    unittest.main()