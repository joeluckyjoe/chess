# gnn_agent/rl_loop/bayesian_supervisor.py

import pandas as pd
import numpy as np
from gnn_agent.rl_loop.reference_detector import Bocd, NormalUnknownMean
import os

class BayesianSupervisor:
    """
    A supervisor that uses a PROVEN, reference implementation of the 
    Bayesian Changepoint Detector.
    """
    def __init__(self, probability_threshold: float = 0.5, min_data_points: int = 40, hazard_rate: float = 0.1):
        self.prob_threshold = probability_threshold
        self.min_data_points = min_data_points
        if hazard_rate == 0: hazard_rate = 0.001
        self.hazard_lambda = 1 / hazard_rate
        self.detector = None

    def check_stagnation_from_log(self, log_file_path: str) -> bool:
        """
        Analyzes a training log file for stagnation.
        """
        try:
            if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0: return False
            df = pd.read_csv(log_file_path)
            if 'policy_loss' not in df.columns or df.empty: return False
            policy_losses = df['policy_loss'].dropna().values
        except (pd.errors.EmptyDataError, KeyError): return False

        if len(policy_losses) < self.min_data_points:
            return False

        # --- THE FIX: Initialize the prior belief with the first data point, not a hardcoded 0. ---
        initial_mean_belief = policy_losses[0]
        recent_variance = np.var(policy_losses[-self.min_data_points:])
        if recent_variance == 0: recent_variance = 0.001
        
        # This reference implementation has a clear way to define the statistical model.
        # We now provide a sensible starting belief for the mean.
        model = NormalUnknownMean(mu=initial_mean_belief, kappa=1, alpha=1, beta=recent_variance)
        self.detector = Bocd(model=model, hazard_lambda=self.hazard_lambda)

        for i, loss in enumerate(policy_losses):
            self.detector.update(loss)
            
            if i >= self.min_data_points:
                map_run_length = np.argmax(self.detector.R)
                
                if map_run_length == 0:
                    map_probability = self.detector.R[map_run_length]
                    if map_probability > self.prob_threshold:
                        print(f"INFO: Stagnation detected at game {i}. MAP Estimate for run length was 0 with probability {map_probability:.4f}.")
                        return True
        
        return False