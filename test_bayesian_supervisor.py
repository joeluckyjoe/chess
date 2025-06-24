import unittest
import os
import pandas as pd
import numpy as np

# Adjust the path to correctly import the BayesianSupervisor from the gnn_agent package.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gnn_agent.rl_loop.bayesian_supervisor import BayesianSupervisor


class TestBayesianSupervisor(unittest.TestCase):
    """
    Unit tests for the BayesianSupervisor class.
    """
    def setUp(self):
        """Set up for each test. This runs before every test method."""
        self.log_file_path = 'test_loss_log_bayesian.csv'
        self.config = {
            'SUPERVISOR_BAYESIAN_PENALTY': 3,
            'SUPERVISOR_WINDOW_SIZE': 20, # This sets min_data_points to 40
            'SUPERVISOR_PERFORMANCE_THRESHOLD': 7.0
        }
        self.supervisor = BayesianSupervisor(self.config)
        self.min_data_points = self.config['SUPERVISOR_WINDOW_SIZE'] * 2

    def tearDown(self):
        """Clean up after each test. This runs after every test method."""
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

    def _create_log_file(self, data):
        """Helper function to create a CSV log file with policy losses."""
        df = pd.DataFrame({'policy_loss': data})
        df.to_csv(self.log_file_path, index=False)

    def test_no_log_file_or_empty(self):
        """Test that it returns False if the log file doesn't exist or is empty."""
        self.assertFalse(self.supervisor.check_for_stagnation('non_existent_file.csv'))
        self._create_log_file([])
        with open(self.log_file_path, 'w') as f:
            f.write('')
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_insufficient_data(self):
        """Test that it returns False if there aren't enough data points."""
        # We need 40 data points, so 39 should be insufficient.
        data = np.random.rand(self.min_data_points - 1)
        self._create_log_file(data)
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_safety_net_triggered(self):
        """Test that the safety net is triggered by catastrophic performance."""
        # Recent losses are very high, well above the 7.0 threshold.
        data = np.full(self.min_data_points, 8.0)
        self._create_log_file(data)
        self.assertTrue(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_stagnation_detected_by_bcpd(self):
        """Test that stagnation is detected when performance gets worse after a changepoint."""
        # Create a clear changepoint where loss increases
        segment1 = np.random.normal(loc=1.0, scale=0.1, size=30)
        segment2 = np.random.normal(loc=2.0, scale=0.1, size=30) # Mean clearly increases
        all_data = np.concatenate([segment1, segment2])
        self._create_log_file(all_data)
        self.assertTrue(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_improvement_is_not_stagnation(self):
        """Test that no stagnation is detected when performance improves after a changepoint."""
        # Create a clear changepoint where loss decreases
        segment1 = np.random.normal(loc=2.0, scale=0.1, size=30)
        segment2 = np.random.normal(loc=1.0, scale=0.1, size=30) # Mean clearly decreases
        all_data = np.concatenate([segment1, segment2])
        self._create_log_file(all_data)
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_no_changepoint_no_trigger(self):
        """Test that no stagnation is detected when there is a steady trend (no changepoint)."""
        # Create data with a steady downward trend
        data = np.linspace(3.0, 1.0, num=self.min_data_points)
        self._create_log_file(data)
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))


if __name__ == '__main__':
    # Use the newer TestLoader method to avoid deprecation warnings
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBayesianSupervisor)
    runner = unittest.TextTestRunner()
    runner.run(suite)
