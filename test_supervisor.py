import unittest
import os
import pandas as pd
import numpy as np

# As we are in a different directory for testing, we need to adjust the path
# to correctly import the StatisticalSupervisor.
import sys
# Add the parent directory (where gnn_agent is) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gnn_agent.rl_loop.statistical_supervisor import StatisticalSupervisor


class TestStatisticalSupervisor(unittest.TestCase):
    """
    Unit tests for the StatisticalSupervisor class.
    """
    def setUp(self):
        """
        Set up for each test. This runs before every test method.
        """
        self.log_file_path = 'test_loss_log.csv'
        self.config = {
            'SUPERVISOR_P_VALUE_THRESHOLD': 0.05,
            'SUPERVISOR_WINDOW_SIZE': 10,
            'SUPERVISOR_PERFORMANCE_THRESHOLD': 1.8
        }
        self.supervisor = StatisticalSupervisor(self.config)
        # min_data_points will be 10 * 2 = 20
        self.min_data_points = self.config['SUPERVISOR_WINDOW_SIZE'] * 2

    def tearDown(self):
        """
        Clean up after each test. This runs after every test method.
        """
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)

    def _create_log_file(self, data):
        """Helper function to create a CSV log file with policy losses."""
        df = pd.DataFrame({'policy_loss': data})
        df.to_csv(self.log_file_path, index=False)

    def test_no_log_file(self):
        """
        Test that it returns False if the log file does not exist.
        """
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_empty_log_file(self):
        """
        Test that it returns False if the log file is empty.
        """
        self._create_log_file([])
        # Make the file truly empty
        with open(self.log_file_path, 'w') as f:
            f.write('')
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_insufficient_data(self):
        """
        Test that it returns False if there are not enough data points.
        """
        # We need 20 data points, so 19 should be insufficient.
        data = np.random.rand(self.min_data_points - 1)
        self._create_log_file(data)
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_safety_net_triggered(self):
        """
        Test that the safety net is triggered by catastrophic performance.
        """
        # Baseline losses are good
        baseline_data = np.full(self.config['SUPERVISOR_WINDOW_SIZE'], 1.0)
        # Recent losses are very high, well above the 1.8 threshold
        recent_data = np.full(self.config['SUPERVISOR_WINDOW_SIZE'], 2.5)
        
        all_data = np.concatenate([baseline_data, recent_data])
        self._create_log_file(all_data)
        
        self.assertTrue(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_stagnation_detected_by_ttest(self):
        """
        Test that stagnation is detected when recent performance is not improving.
        """
        # --- THIS IS THE CORRECTED TEST ---
        # We now use np.random.normal to simulate noisy, realistic loss data.
        
        # Baseline data: 10 points with a mean of ~1.0
        baseline_mean = 1.0
        baseline_data = np.random.normal(loc=baseline_mean, scale=0.1, size=self.config['SUPERVISOR_WINDOW_SIZE'])
        
        # Recent data: 10 points with a mean of ~1.15 (slight worsening / stagnation)
        recent_mean = 1.15 
        recent_data = np.random.normal(loc=recent_mean, scale=0.1, size=self.config['SUPERVISOR_WINDOW_SIZE'])
        
        # Ensure average of recent data is well below the safety net threshold of 1.8
        self.assertTrue(np.mean(recent_data) < self.config['SUPERVISOR_PERFORMANCE_THRESHOLD'])
        
        all_data = np.concatenate([baseline_data, recent_data])
        self._create_log_file(all_data)
        
        # With the recent mean being statistically higher, we expect the supervisor to trigger.
        self.assertTrue(self.supervisor.check_for_stagnation(self.log_file_path))
        
    def test_clear_improvement_no_trigger(self):
        """
        Test that no stagnation is detected when performance is clearly improving.
        """
        # Baseline losses are high
        baseline_data = np.random.normal(loc=1.5, scale=0.1, size=self.config['SUPERVISOR_WINDOW_SIZE'])
        # Recent losses are significantly lower
        recent_data = np.random.normal(loc=0.8, scale=0.1, size=self.config['SUPERVISOR_WINDOW_SIZE'])
        
        all_data = np.concatenate([baseline_data, recent_data])
        self._create_log_file(all_data)
        
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))

    def test_stable_good_performance_no_trigger(self):
        """
        Test that no stagnation is detected during stable, good performance.
        """
        # Baseline and recent losses are both low and stable
        baseline_data = np.random.normal(loc=0.75, scale=0.05, size=self.config['SUPERVISOR_WINDOW_SIZE'])
        recent_data = np.random.normal(loc=0.70, scale=0.05, size=self.config['SUPERVISOR_WINDOW_SIZE'])
        
        all_data = np.concatenate([baseline_data, recent_data])
        self._create_log_file(all_data)
        
        self.assertFalse(self.supervisor.check_for_stagnation(self.log_file_path))


if __name__ == '__main__':
    # We use a try-finally block to ensure tearDown is called to clean up the test file
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestStatisticalSupervisor))
    runner = unittest.TextTestRunner()
    
    # We create an instance to call tearDown
    test_instance = TestStatisticalSupervisor()
    test_instance.setUp()
    try:
        runner.run(suite)
    finally:
        # Explicitly clean up the file in case of interruption
        test_instance.tearDown()