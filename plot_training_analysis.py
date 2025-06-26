import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ruptures as rpt
from pathlib import Path
import sys

# Add project root to path to allow importing from config
# This ensures that the script can find the config module
project_root_for_imports = Path(os.path.abspath(__file__)).parent.parent
if str(project_root_for_imports) not in sys.path:
    sys.path.insert(0, str(project_root_for_imports))

from config import get_paths

# --- Configuration ---
# You can adjust this penalty value to tune the sensitivity of the changepoint detection.
CHANGEPOINT_PENALTY = 2 
OUTPUT_FILENAME = "supervisor_analysis_plot.png"

def plot_supervisor_analysis(loss_log_path, output_filename):
    """
    Loads training loss data, performs changepoint analysis, and generates a comprehensive plot.
    """
    try:
        df = pd.read_csv(loss_log_path)
        print(f"Successfully loaded loss data from '{loss_log_path.name}' with {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: Loss log file not found at '{loss_log_path}'")
        return

    # --- Data Preparation ---
    df = df.sort_values('game').reset_index(drop=True)
    rolling_window = 10
    df['policy_loss_ma'] = df['policy_loss'].rolling(window=rolling_window, min_periods=1).mean()
    df['value_loss_ma'] = df['value_loss'].rolling(window=rolling_window, min_periods=1).mean()

    # --- Changepoint Detection ---
    self_play_data = df[df['game_type'] == 'self-play']['policy_loss_ma'].dropna()
    
    changepoints = []
    if len(self_play_data) > 1:
        print(f"Running changepoint detection with penalty={CHANGEPOINT_PENALTY}...")
        algo = rpt.Pelt(model="rbf").fit(self_play_data.values)
        try:
            result = algo.predict(pen=CHANGEPOINT_PENALTY)
            # Ensure we don't try to access indices that are out of bounds
            valid_indices = [idx for idx in result[:-1] if idx < len(self_play_data)]
            changepoint_indices = self_play_data.index[valid_indices]
            changepoints = df['game'].iloc[changepoint_indices].tolist()
            print(f"Detected {len(changepoints)} changepoints at or near game numbers: {changepoints}")
        except Exception as e:
            print(f"Could not predict changepoints: {e}")

    # --- Plotting ---
    print("Generating plot...")
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Plot Policy Loss
    ax1.plot(df['game'], df['policy_loss_ma'], color='deepskyblue', label=f'Policy Loss ({rolling_window}-game MA)', zorder=5)
    ax1.set_xlabel('Game Number', fontsize=14)
    ax1.set_ylabel('Policy Loss (Smoothed)', color='deepskyblue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='deepskyblue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot Value Loss on a secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df['game'], df['value_loss_ma'], color='salmon', linestyle='--', label=f'Value Loss ({rolling_window}-game MA)', zorder=5)
    ax2.set_ylabel('Value Loss (Smoothed)', color='salmon', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='salmon')

    # Add shaded regions for training modes
    last_game = df['game'].max()
    color = 'lightcyan' # Default color
    for i in range(len(df) - 1):
        start_game, end_game = df['game'].iloc[i], df['game'].iloc[i+1]
        game_type = df['game_type'].iloc[i]
        color = 'lightcyan' if game_type == 'self-play' else 'lightgreen'
        ax1.axvspan(start_game, end_game, facecolor=color, alpha=0.3, zorder=0)

    # Fill the last segment
    if len(df) > 0:
        last_game_type = df['game_type'].iloc[-1]
        last_color = 'lightcyan' if last_game_type == 'self-play' else 'lightgreen'
        ax1.axvspan(df['game'].iloc[-1], last_game, facecolor=last_color, alpha=0.3, zorder=0)

    # Add vertical lines for detected changepoints
    for cp in changepoints:
        ax1.axvline(x=cp, color='darkviolet', linestyle=':', linewidth=2, label=f'Detected Changepoint', zorder=10)
    
    # --- Final Touches ---
    fig.suptitle('Supervisor Training Analysis', fontsize=20, weight='bold')
    ax1.set_title(f'Policy/Value Loss vs. Game Number (up to game {int(last_game)})', fontsize=16)
    
    # Create a single, clean legend
    from matplotlib.patches import Patch
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    legend_elements = lines + lines2 + [
        Patch(facecolor='lightcyan', alpha=0.5, label='Self-Play Mode'),
        Patch(facecolor='lightgreen', alpha=0.5, label='Mentor Mode')
    ]
    
    unique_labels = {}
    # Use a direct mapping to avoid issues with different line objects for the same label
    for handle in legend_elements:
        label = handle.get_label()
        if label not in unique_labels:
            unique_labels[label] = handle
            
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(0.9, 0.88))
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    plt.savefig(output_filename, dpi=300)
    print(f"\nAnalysis plot saved to: {output_filename}")
    plt.close()


if __name__ == '__main__':
    # Use the centralized get_paths function to find all relevant paths
    paths = get_paths()
    
    # --- DYNAMICALLY FIND THE CORRECT LOG FILE ---
    # The main training script saves the log file in the main data directory.
    # We determine the data directory based on the environment (Colab vs. local).
    if 'COLAB_GPU' in os.environ:
        # The parent of the 'checkpoints' directory is the main data root on Drive
        data_root = paths.checkpoints_dir.parent
    else:
        # For local runs, assume it's in the project root with the script.
        data_root = paths.project_root

    # Search for all files matching the pattern 'loss_log*.csv'
    log_files = list(data_root.glob('loss_log*.csv'))
    
    if not log_files:
        print(f"Error: Could not find any 'loss_log*.csv' files in '{data_root}'")
        print("Please ensure 'run_training.py' has been run and has generated a loss log file.")
    else:
        # Sort by modification time to find the most recent log file
        latest_log_file = max(log_files, key=os.path.getmtime)
        print(f"Found {len(log_files)} log file(s). Using the most recent: '{latest_log_file.name}'")
        plot_supervisor_analysis(latest_log_file, OUTPUT_FILENAME)
