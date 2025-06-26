# File: gnn_agent/rl_loop/mentor_trigger.py (FINAL, CORRECTED)

import numpy as np
from functools import partial
from scipy.stats import linregress
import bayesian_changepoint_detection.online_changepoint_detection as bcpd_utils

class AdaptiveBCPDMonitor:
    """
    A stateful, online monitor that uses an adaptive Bayesian Change Point
    Detection algorithm to classify the regime of a time-series data stream.
    The regime analysis is performed using a fast and robust linear regression.
    """
    def __init__(self,
                 min_analysis_window=20,
                 improving_hazard=1000,
                 plateau_hazard=50,
                 worsening_hazard=10,
                 p_value_threshold=0.05):
        """
        Initializes the state of the monitor.
        """
        print("Initializing AdaptiveBCPDMonitor (using SciPy)...")
        self.min_window = min_analysis_window
        self.hazard_rates = {
            'Improving': improving_hazard,
            'Plateau': plateau_hazard,
            'Worsening': worsening_hazard,
            'Pending': 250
        }
        self.p_value_threshold = p_value_threshold

        self.observation_model = bcpd_utils.StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
        self.reset()

    def reset(self):
        """Resets the monitor's state, typically after a mode switch."""
        print("BCPD Monitor has been reset.")
        self.observation_model = bcpd_utils.StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
        self.current_R_row = np.array([1.0])
        self.current_run_data = []
        self.time_step = 0
        self.current_regime = 'Pending'

    def _analyze_regime(self, subsample):
        """
        Private method to analyze a subsample of data with Scipy's linregress.
        """
        if len(subsample) < 2:
            return 'Pending'
            
        time_steps = np.arange(len(subsample))
        
        try:
            slope, intercept, r_value, p_value, std_err = linregress(time_steps, subsample)
        except (ValueError, IndexError):
            return 'Plateau'

        if slope < 0 and p_value < self.p_value_threshold:
            return 'Improving'
        elif slope > 0 and p_value < self.p_value_threshold:
            return 'Worsening'
        else:
            return 'Plateau'

    def update(self, new_data_point):
        """
        Processes a single new data point and updates the monitor's state.
        """
        current_run_length = self.current_R_row.argmax()

        if current_run_length == 0 and self.time_step > 0:
            self.current_run_data = []

        self.current_run_data.append(new_data_point)
        
        if len(self.current_run_data) > self.min_window:
            self.current_regime = self._analyze_regime(np.array(self.current_run_data))
        else:
            self.current_regime = 'Pending'
        
        # This is the simple, correct implementation using the tools we know exist.
        hazard_rate = self.hazard_rates[self.current_regime]
        hazard_function = partial(bcpd_utils.constant_hazard, hazard_rate)

        pred_probs = self.observation_model.pdf(new_data_point)
        
        next_R_row = np.zeros(self.time_step + 2)
        
        next_R_row[1:] = self.current_R_row * pred_probs * (1 - hazard_function(np.arange(self.time_step + 1)))
        
        next_R_row[0] = np.sum(self.current_R_row * pred_probs * hazard_function(np.arange(self.time_step + 1)))
        
        total_prob = np.sum(next_R_row)
        if total_prob == 0:
            total_prob = 1e-9

        next_R_row /= total_prob
        
        self.current_R_row = next_R_row
        self.observation_model.update_theta(new_data_point)
        self.time_step += 1
        
        return self.current_regime