# FILENAME: gnn_agent/rl_loop/bayesian_supervisor.py
import os
import pandas as pd
import numpy as np
import ruptures as rpt
from pathlib import Path

class BayesianSupervisor:
    """
    A supervisor that uses Bayesian Changepoint Detection to identify RECENT
    performance plateaus, and also includes a safety net.
    """
    def __init__(self, config):
        """
        Initializes the supervisor using a configuration dictionary.
        """
        self.penalty = config.get('SUPERVISOR_BAYESIAN_PENALTY', 0.8)
        self.min_data_points = config.get('SUPERVISOR_WINDOW_SIZE', 20)
        self.performance_threshold = config.get('SUPERVISOR_PERFORMANCE_THRESHOLD', 7.0)
        
        # Only care about changepoints that happened in the last N games.
        self.recency_window = config.get('SUPERVISOR_RECENCY_WINDOW', 50)
        
        print("BayesianSupervisor initialized.")
        print(f"  - Stagnation Plateau Check: BCPD (model=rbf, penalty={self.penalty})")
        print(f"  - Recency Window: {self.recency_window} games")
        print(f"  - Safety Net Check: avg_loss > {self.performance_threshold}")


    def check_for_stagnation(self, log_file_path: str) -> bool:
        """
        Analyzes a training log file for stagnation using BCPD, but only
        considers recent changepoints.
        """
        try:
            if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
                return False
            df = pd.read_csv(log_file_path)
            if 'policy_loss' not in df.columns or df.empty:
                return False
            
            # Continue to smooth the entire series first
            policy_losses = df['policy_loss'].rolling(window=5, min_periods=1).mean().dropna().values
        
        except (pd.errors.EmptyDataError, KeyError):
            return False

        # --- BUG FIX: Check for overall data length before slicing ---
        # Ensure we have at least a minimum number of total data points to work with.
        if len(policy_losses) < self.min_data_points:
            return False

        # --- 1. Safety Net Check (operates on the most recent data) ---
        recent_losses = policy_losses[-self.recency_window:]
        current_performance = np.mean(recent_losses)
        
        if current_performance > self.performance_threshold:
            print(f"SUPERVISOR (Safety Net): Performance threshold breached ({current_performance:.3f} > {self.performance_threshold}). Switching to MENTOR mode.")
            return True

        # --- 2. Stagnation Plateau Check (operates ONLY on the recent window) ---

        # --- BUG FIX: The BCPD analysis now runs on the SLICE, not the full history. ---
        if len(recent_losses) < self.min_data_points:
            # Not enough *recent* data points for a meaningful BCPD analysis
            return False

        try:
            algo = rpt.Pelt(model="rbf").fit(recent_losses)
            changepoints = algo.predict(pen=self.penalty)
        except Exception as e:
            print(f"[Warning] BCPD (RBF model) failed with error: {e}. Defaulting to no trigger.")
            return False

        if len(changepoints) <= 1:
            return False
        
        # This index is now correctly relative to the `recent_losses` array
        last_changepoint_idx = changepoints[-2]
        
        # --- BUG FIX: The old, incorrect recency check is now removed. ---
        # The logic is inherently recent because we are analyzing a slice of data.
        
        segment_after = recent_losses[last_changepoint_idx:]
        
        if len(segment_after) < 10: # Ensure the segment after the changepoint is substantial enough
             return False

        segment_before = recent_losses[last_changepoint_idx - len(segment_after) : last_changepoint_idx]
        
        if len(segment_before) == 0:
            return False

        mean_after = np.mean(segment_after)
        mean_before = np.mean(segment_before)

        if mean_after > mean_before:
            # --- BUG FIX: Updated log message for clarity ---
            print(f"SUPERVISOR (BCPD): RECENT stagnation detected. Mean loss increased from {mean_before:.4f} to {mean_after:.4f} after changepoint at index {last_changepoint_idx} (within the last {self.recency_window} games).")
            return True

        return False