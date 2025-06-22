# visualize_detector.py

import matplotlib
matplotlib.use('Agg') # Explicitly use the non-interactive Agg backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from gnn_agent.rl_loop.bayesian_detector import BayesianChangepointDetector

def generate_stagnation_data(n_segment1=50, n_segment2=50, mean1=0.8, mean2=0.2, variance=0.05):
    """Generates time series data with a single, sharp changepoint."""
    np.random.seed(42) # for reproducibility
    segment1 = np.random.normal(mean1, np.sqrt(variance), n_segment1)
    segment2 = np.random.normal(mean2, np.sqrt(variance), n_segment2)
    return np.concatenate((segment1, segment2))

def main():
    """
    Main function to run the visualization.
    """
    # --- 1. Setup and Data Generation ---
    output_filename = "detector_visualization.png"
    data_variance = 0.05
    hazard_rate = 0.1
    data = generate_stagnation_data(variance=data_variance)
    changepoint_location = 50

    # --- 2. Instantiate and Run the Detector ---
    detector = BayesianChangepointDetector(hazard_rate=hazard_rate, data_variance=data_variance)

    snapshots = {}
    snapshot_points = [changepoint_location - 1, changepoint_location, changepoint_location + 5, changepoint_location + 20, len(data) -1]

    for i, data_point in enumerate(data):
        detector.update(data_point)
        if i in snapshot_points:
            snapshots[i] = np.copy(detector.R)

    # --- 3. Plotting ---
    fig, axes = plt.subplots(len(snapshot_points) + 1, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2] + [1]*len(snapshot_points)})
    fig.suptitle("Evolution of Detector's Belief State", fontsize=16)

    ax_data = axes[0]
    ax_data.plot(data)
    ax_data.axvline(x=changepoint_location, color='r', linestyle='--', label=f'True Changepoint (t={changepoint_location})')
    ax_data.set_title("Input Data Series")
    ax_data.set_xlabel("Time Step")
    ax_data.set_ylabel("Value")
    ax_data.legend()
    ax_data.grid(True, alpha=0.5)

    for i, t in enumerate(snapshot_points):
        ax = axes[i+1]
        run_lengths = np.arange(len(snapshots[t]))
        ax.bar(run_lengths, snapshots[t], width=1.0)
        ax.set_title(f"Belief State at t={t}")
        ax.set_xlabel("Run Length")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.5)
        ax.set_xlim(-1, max(run_lengths) + 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- CHANGE: Save the figure to a file instead of showing it ---
    plt.savefig(output_filename)
    plt.close(fig) # Close the figure to free up memory
    
    print(f"Plot successfully saved to {output_filename}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")