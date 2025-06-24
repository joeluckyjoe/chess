import os
import pandas as pd
import numpy as np
import ruptures as rpt
from pathlib import Path

class BayesianSupervisor:
    """
    A supervisor that uses Bayesian Changepoint Detection (via the 'ruptures'
    library) to identify performance plateaus or regressions, and also includes
    a safety net to detect catastrophic performance drops.
    """
    def __init__(self, config):
        """
        Initializes the supervisor using a configuration dictionary.

        Args:
            config (dict): A dictionary containing all supervisor parameters.
        """
        # --- Changepoint Detection Parameters ---
        self.penalty = config.get('SUPERVISOR_BAYESIAN_PENALTY', 1) 
        self.min_data_points = config.get('SUPERVISOR_WINDOW_SIZE', 20) * 2

        # --- Performance (Safety Net) Parameters ---
        self.performance_threshold = config.get('SUPERVISOR_PERFORMANCE_THRESHOLD', 7.0)
        
        print("BayesianSupervisor initialized.")
        # CORRECTED: Switched to the more powerful 'rbf' model.
        print(f"  - Stagnation Plateau Check: BCPD (model=rbf, penalty={self.penalty})")
        print(f"  - Safety Net Check: avg_loss > {self.performance_threshold}")


    def check_for_stagnation(self, log_file_path: str) -> bool:
        """
        Analyzes a training log file for stagnation using BCPD.

        Args:
            log_file_path (str): The path to the loss_log_v2.csv file.

        Returns:
            bool: True if stagnation/failure is detected, False otherwise.
        """
        try:
            if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
                return False
            df = pd.read_csv(log_file_path)
            if 'policy_loss' not in df.columns or df.empty:
                return False
            
            policy_losses = df['policy_loss'].rolling(window=5, min_periods=1).mean().dropna().values
        
        except (pd.errors.EmptyDataError, KeyError):
            return False

        if len(policy_losses) < self.min_data_points:
            return False

        # --- 1. Safety Net Check ---
        recent_window_for_perf = policy_losses[-self.min_data_points//2:]
        current_performance = np.mean(recent_window_for_perf)
        
        if current_performance > self.performance_threshold:
            print(f"SUPERVISOR (Safety Net): Performance threshold breached ({current_performance:.3f} > {self.performance_threshold}). Switching to MENTOR mode.")
            return True

        # --- 2. Stagnation Plateau Check (BCPD with RBF model) ---
        # CORRECTED: Switched from the default 'l2' model to 'rbf' for more flexibility.
        try:
            algo = rpt.Pelt(model="rbf").fit(policy_losses)
            changepoints = algo.predict(pen=self.penalty)
        except Exception as e:
            # The RBF model can sometimes fail on certain data patterns.
            # We will log the error and default to no stagnation.
            print(f"[Warning] BCPD (RBF model) failed with error: {e}. Defaulting to no trigger.")
            return False


        if len(changepoints) <= 1:
            return False
        
        last_changepoint_idx = changepoints[-2]
        
        segment_after = policy_losses[last_changepoint_idx:]
        
        if len(segment_after) < 10:
             return False

        segment_before = policy_losses[last_changepoint_idx - len(segment_after) : last_changepoint_idx]
        
        if len(segment_before) == 0:
            return False

        mean_after = np.mean(segment_after)
        mean_before = np.mean(segment_before)

        if mean_after > mean_before:
            print(f"SUPERVISOR (BCPD): Stagnation detected. Mean loss increased from {mean_before:.4f} to {mean_after:.4f} after changepoint at game index {last_changepoint_idx}.")
            return True

        return False
