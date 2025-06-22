# gnn_agent/rl_loop/statistical_supervisor.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import os

class StatisticalSupervisor:
    """
    A supervisor that detects stagnation plateaus using a statistical t-test,
    and also includes a safety net to detect catastrophic performance drops.
    """
    def __init__(self, config):
        """
        Initializes the supervisor using a configuration dictionary.

        Args:
            config (dict): A dictionary containing all supervisor parameters.
        """
        # --- T-Test (Stagnation Plateau) Parameters ---
        self.p_value_threshold = config.get('SUPERVISOR_P_VALUE_THRESHOLD', 0.05)
        self.window_size = config.get('SUPERVISOR_WINDOW_SIZE', 20)
        
        # --- Performance (Safety Net) Parameters ---
        self.performance_threshold = config.get('SUPERVISOR_PERFORMANCE_THRESHOLD', 1.8)
        
        self.min_data_points = self.window_size * 2
        print("StatisticalSupervisor initialized.")
        print(f"  - Stagnation Plateau Check: p-value < {self.p_value_threshold} (window_size={self.window_size})")
        print(f"  - Safety Net Check: avg_loss > {self.performance_threshold}")


    def check_for_stagnation(self, log_file_path: str) -> bool:
        """
        Analyzes a training log file for stagnation or catastrophic performance.

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
            policy_losses = df['policy_loss'].dropna().values
        except (pd.errors.EmptyDataError, KeyError):
            return False

        if len(policy_losses) < self.min_data_points:
            return False

        # --- 1. Safety Net Check ---
        # First, check for catastrophic performance. This is a simple, fast check.
        recent_window_for_perf = policy_losses[-self.window_size:]
        current_performance = np.mean(recent_window_for_perf)
        
        if current_performance > self.performance_threshold:
            print(f"SUPERVISOR (Safety Net): Performance threshold breached ({current_performance:.3f} > {self.performance_threshold}). Switching to MENTOR mode.")
            return True

        # --- 2. Stagnation Plateau Check (T-Test) ---
        # If performance is acceptable, check for a learning plateau.
        recent_window = policy_losses[-self.window_size:]
        baseline_window = policy_losses[-self.min_data_points:-self.window_size]
        
        # We test if the mean of the recent window is significantly GREATER
        # than the mean of the baseline. A low p-value suggests the loss
        # is no longer decreasing.
        _, p_value = ttest_ind(
            recent_window, 
            baseline_window, 
            equal_var=False,         # Welch's t-test, robust to unequal variances
            alternative='greater'  # One-sided test
        )

        if p_value < self.p_value_threshold:
            mean_recent = np.mean(recent_window)
            mean_baseline = np.mean(baseline_window)
            print(f"SUPERVISOR (T-Test): Stagnation plateau detected. Recent performance ({mean_recent:.4f}) is not improving over baseline ({mean_baseline:.4f}). p-value={p_value:.4f}")
            return True

        return False