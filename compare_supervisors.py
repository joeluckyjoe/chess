import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Adjust the path to correctly import supervisors from the gnn_agent package.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from gnn_agent.rl_loop.statistical_supervisor import StatisticalSupervisor
from gnn_agent.rl_loop.bayesian_supervisor import BayesianSupervisor

# --- Configuration ---
# You can adjust the penalty here to see how it affects the Bayesian supervisor's sensitivity
# CORRECTED: Increased penalty to find a balance between sensitivity and stability.
BAYESIAN_PENALTY = 2
LOG_FILE = 'loss_log_v2.csv'
OUTPUT_FILENAME = 'supervisor_comparison_plot.png'

def run_comparison(log_file_path):
    """
    Loads training loss data and compares the decisions of the Statistical
    and Bayesian supervisors over the entire history.
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at '{log_file_path}'")
        return

    print(f"Loading loss data from '{log_file_path}'...")
    df = pd.read_csv(log_file_path)

    # --- Initialize Supervisors ---
    # We use a shared config for base parameters
    config = {
        'SUPERVISOR_WINDOW_SIZE': 20,
        'SUPERVISOR_PERFORMANCE_THRESHOLD': 7.0,
        'SUPERVISOR_P_VALUE_THRESHOLD': 0.05,
        'SUPERVISOR_BAYESIAN_PENALTY': BAYESIAN_PENALTY
    }
    
    statistical_supervisor = StatisticalSupervisor(config)
    bayesian_supervisor = BayesianSupervisor(config)
    
    # --- Simulate Decisions ---
    print("Simulating supervisor decisions over training history...")
    min_games = config['SUPERVISOR_WINDOW_SIZE'] * 2
    
    statistical_triggers = []
    bayesian_triggers = []
    
    # We create a temporary file for each supervisor to use in its check
    temp_log_path = "temp_supervisor_check.csv"

    for i in range(min_games, len(df)):
        # Create a snapshot of the log up to the current game
        df_snapshot = df.iloc[:i]
        df_snapshot.to_csv(temp_log_path, index=False)
        
        # Check statistical supervisor
        if statistical_supervisor.check_for_stagnation(temp_log_path):
            statistical_triggers.append(df['game'].iloc[i])
            
        # Check bayesian supervisor
        if bayesian_supervisor.check_for_stagnation(temp_log_path):
            bayesian_triggers.append(df['game'].iloc[i])

    # Clean up the temporary file
    if os.path.exists(temp_log_path):
        os.remove(temp_log_path)
        
    print(f"Statistical Supervisor would have triggered {len(statistical_triggers)} times.")
    print(f"Bayesian Supervisor (penalty={BAYESIAN_PENALTY}) would have triggered {len(bayesian_triggers)} times.")

    # --- Plotting ---
    print("Generating comparison plot...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(20, 10))

    # Plot the smoothed policy loss
    window_size = 10
    df['policy_loss_smoothed'] = df['policy_loss'].rolling(window=window_size).mean()
    plt.plot(df['game'], df['policy_loss_smoothed'], color='deepskyblue', label=f'Policy Loss ({window_size}-game MA)', zorder=5)

    # Plot vertical lines for each supervisor's triggers
    for game in statistical_triggers:
        plt.axvline(x=game, color='lightgreen', linestyle='--', linewidth=1.5, label='Statistical Trigger', zorder=10)

    for game in bayesian_triggers:
        plt.axvline(x=game, color='darkviolet', linestyle=':', linewidth=2, label='Bayesian Trigger', zorder=10)

    # --- Final Touches ---
    plt.title('Supervisor Decision Comparison', fontsize=20, weight='bold')
    plt.xlabel('Game Number', fontsize=14)
    plt.ylabel('Policy Loss (Smoothed)', fontsize=14)
    
    # Create a clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"\nAnalysis complete. Comparison plot saved to: {OUTPUT_FILENAME}")
    plt.close()


if __name__ == '__main__':
    run_comparison(LOG_FILE)
