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
            'ruptures_penalty': 3  # A standard penalty value is now robust due to normalization
        }
        self.supervisor = TrainingSupervisor(self.config)

    def test_should_switch_to_mentor_logic_with_synthetic_data(self):
        """
        Test the change point detection logic with synthetic data.
        """
        # SCENARIO 1: No stagnation (steady improvement)
        # The derivative of this signal is constant, so its standard deviation is zero.
        improving_losses = np.linspace(start=1.0, stop=0.2, num=40)
        for loss in improving_losses:
            self.supervisor.record_self_play_loss(loss)
        
        self.assertFalse(self.supervisor.should_switch_to_mentor(), "Should not switch with improving losses")

        # SCENARIO 2: Stagnation
        # The derivative changes from negative to zero, creating a non-zero std dev.
        stagnant_part = np.full(10, 0.4)
        improving_part = np.linspace(start=1.0, stop=0.41, num=30)
        stagnant_losses = np.concatenate([improving_part, stagnant_part])
        
        self.supervisor.self_play_policy_losses.clear()
        for loss in stagnant_losses:
            self.supervisor.record_self_play_loss(loss)

        self.assertTrue(self.supervisor.should_switch_to_mentor(), "Should switch with stagnant losses")

# Minimal boilerplate to run tests if the file is executed directly
if __name__ == '__main__':
    unittest.main()