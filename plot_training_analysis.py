import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ruptures as rpt
from pathlib import Path

# --- Configuration ---
# NOTE: This is the most important parameter to tune.
# Lower it to make the detection more sensitive, raise it to make it less sensitive.
CHANGEPOINT_PENALTY = 1 

def get_project_root():
    """
    Determines the project root directory based on the execution environment.
    """
    # Check for a Colab environment variable
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected. Using Google Drive paths.")
        return Path('/content/drive/MyDrive/ChessMCTS_RL')
    else:
        print("Local environment detected.")
        # Assumes the script is run from the project's root directory
        return Path('.')

def plot_supervisor_analysis(loss_log_path, output_filename="supervisor_analysis_plot.png"):
    """
    Loads training loss data, performs changepoint analysis, and generates a comprehensive plot.
    """
    if not loss_log_path.exists():
        print(f"Error: Loss log file not found at '{loss_log_path}'")
        return

    print(f"Loading loss data from '{loss_log_path}'...")
    df = pd.read_csv(loss_log_path)
    
    # --- Data Preparation ---
    # Ensure data is sorted by game number
    df = df.sort_values('game').reset_index(drop=True)

    # Calculate rolling averages to smooth out noise
    rolling_window = 10
    df['policy_loss_ma'] = df['policy_loss'].rolling(window=rolling_window, min_periods=1).mean()
    df['value_loss_ma'] = df['value_loss'].rolling(window=rolling_window, min_periods=1).mean()

    # --- Changepoint Detection ---
    # We only analyze the self-play data for stagnation
    self_play_data = df[df['game_type'] == 'self-play']['policy_loss_ma'].dropna()
    
    changepoints = []
    if len(self_play_data) > 1:
        print(f"Running changepoint detection with penalty={CHANGEPOINT_PENALTY}...")
        # Use Pelt for exact segmentation
        algo = rpt.Pelt(model="l2").fit(self_play_data.values)
        try:
            result = algo.predict(pen=CHANGEPOINT_PENALTY)
            # The result gives indices within the self_play_data Series.
            # We need to map them back to the original 'game' numbers.
            changepoint_indices = self_play_data.index[result[:-1]] # Exclude the end of the series
            changepoints = df['game'].iloc[changepoint_indices].tolist()
            print(f"Detected {len(changepoints)} changepoints at game numbers: {changepoints}")
        except Exception as e:
            print(f"Could not predict changepoints: {e}")

    # --- Plotting ---
    print("Generating plot...")
    fig, ax1 = plt.subplots(figsize=(18, 9))

    # Plot Policy Loss on the primary Y-axis
    ax1.plot(df['game'], df['policy_loss_ma'], color='deepskyblue', label=f'Policy Loss ({rolling_window}-game MA)', zorder=5)
    ax1.set_xlabel('Game Number', fontsize=14)
    ax1.set_ylabel('Policy Loss (Smoothed)', color='deepskyblue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='deepskyblue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a secondary Y-axis for Value Loss
    ax2 = ax1.twinx()
    ax2.plot(df['game'], df['value_loss_ma'], color='salmon', linestyle='--', label=f'Value Loss ({rolling_window}-game MA)', zorder=5)
    ax2.set_ylabel('Value Loss (Smoothed)', color='salmon', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='salmon')

    # --- Add Shaded Regions for Training Mode ---
    last_game = df['game'].max()
    for i in range(len(df) - 1):
        start_game = df['game'].iloc[i]
        end_game = df['game'].iloc[i+1]
        game_type = df['game_type'].iloc[i]
        color = 'lightcyan' if game_type == 'self-play' else 'lightgreen'
        ax1.axvspan(start_game, end_game, facecolor=color, alpha=0.3, zorder=0)

    # Fill the last segment
    if len(df) > 0:
        ax1.axvspan(df['game'].iloc[-1], last_game, facecolor=color, alpha=0.3, zorder=0)


    # --- Add Vertical Lines for Changepoints ---
    for cp in changepoints:
        ax1.axvline(x=cp, color='red', linestyle='-.', linewidth=1.5, label=f'Changepoint (Penalty={CHANGEPOINT_PENALTY})', zorder=10)
    
    # --- Final Touches ---
    fig.suptitle('Supervisor Training Analysis', fontsize=20, weight='bold')
    ax1.set_title(f'Policy/Value Loss vs. Game Number (up to game {last_game})', fontsize=16)
    
    # Create a single legend for all plot elements
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Add custom patches for the background colors
    from matplotlib.patches import Patch
    legend_elements = lines + lines2 + [
        Patch(facecolor='lightcyan', alpha=0.5, label='Self-Play Mode'),
        Patch(facecolor='lightgreen', alpha=0.5, label='Mentor Mode')
    ]
    
    # Remove duplicate changepoint labels from legend
    unique_labels = {}
    for line, label in zip(legend_elements, [l.get_label() for l in legend_elements]):
        if label not in unique_labels:
            unique_labels[label] = line
            
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(0.9, 0.88))
    
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle

    # Save the plot
    output_path = get_project_root() / output_filename
    plt.savefig(output_path, dpi=300)
    print(f"\nAnalysis plot saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    project_root = get_project_root()
    loss_csv_file = project_root / 'loss_log_v2.csv'
    plot_supervisor_analysis(loss_csv_file)

