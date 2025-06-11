# tests/test_rl_loop/test_training_supervisor.py

import unittest
import numpy as np
import collections
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
            'supervisor_loss_history_size': 40,
            'stagnation_window': 0.25,
            'ruptures_model': 'l2',
            'ruptures_penalty': 3,
            'mentor_history_size': 3,
            'mentor_win_threshold': 1,
            'mentor_draw_threshold': 2
        }
        self.supervisor = TrainingSupervisor(self.config)

    def test_initialization(self):
        """
        Test that the supervisor initializes all parameters correctly.
        """
        self.assertIsNotNone(self.supervisor)
        self.assertEqual(self.supervisor.loss_history_size, 40)
        self.assertEqual(self.supervisor.mentor_history_size, 3)
        self.assertEqual(self.supervisor.mentor_win_threshold, 1)
        self.assertEqual(self.supervisor.mentor_draw_threshold, 2)

    def test_should_switch_to_mentor_logic_with_synthetic_data(self):
        """
        Test the change point detection logic with synthetic data.
        """
        # SCENARIO 1: No stagnation (steady improvement)
        improving_losses = np.linspace(start=1.0, stop=0.2, num=40)
        for loss in improving_losses:
            self.supervisor.record_self_play_loss(loss)
        
        self.assertFalse(self.supervisor.should_switch_to_mentor(), "Should not switch with improving losses")

        # SCENARIO 2: Stagnation
        stagnant_part = np.full(10, 0.4)
        improving_part = np.linspace(start=1.0, stop=0.41, num=30)
        stagnant_losses = np.concatenate([improving_part, stagnant_part])
        
        self.supervisor.self_play_policy_losses.clear()
        for loss in stagnant_losses:
            self.supervisor.record_self_play_loss(loss)

        self.assertTrue(self.supervisor.should_switch_to_mentor(), "Should switch with stagnant losses")

    def test_should_switch_to_self_play_history_logic(self):
        """
        Test the logic for switching back to self-play based on mentor game history.
        """
        # Case 1: Not enough games played yet
        self.supervisor.record_mentor_game_outcome(0.5) # One draw
        self.assertFalse(self.supervisor.should_switch_to_self_play(), "Should not switch with incomplete history")

        # Case 2: History is full, but thresholds are not met (1 draw, 2 losses)
        self.supervisor.record_mentor_game_outcome(0.0)
        self.supervisor.record_mentor_game_outcome(0.0)
        self.assertFalse(self.supervisor.should_switch_to_self_play(), "Should not switch if thresholds not met")

        # Case 3: Draw threshold is met (2 draws, 1 loss)
        self.supervisor.record_mentor_game_outcome(0.5) # History is now [0.0, 0.0, 0.5] -> 1 draw
        self.supervisor.record_mentor_game_outcome(0.5) # History is now [0.0, 0.5, 0.5] -> 2 draws
        self.assertTrue(self.supervisor.should_switch_to_self_play(), "Should switch when draw threshold is met")

        # Case 4: Win threshold is met (1 win, 2 losses)
        self.supervisor.mentor_game_outcomes.clear()
        self.supervisor.record_mentor_game_outcome(0.0)
        self.supervisor.record_mentor_game_outcome(0.0)
        self.supervisor.record_mentor_game_outcome(1.0)
        self.assertTrue(self.supervisor.should_switch_to_self_play(), "Should switch when win threshold is met")
        
        # Verify that histories are cleared after a positive decision
        self.assertEqual(len(self.supervisor.mentor_game_outcomes), 0, "Mentor history should be cleared after switching")

# Minimal boilerplate to run tests if the file is executed directly
if __name__ == '__main__':
    unittest.main()