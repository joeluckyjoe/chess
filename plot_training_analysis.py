import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ruptures as rpt
from pathlib import Path
import sys

# Add project root to path to allow importing from config
project_root_for_imports = Path(os.path.abspath(__file__)).parent
if str(project_root_for_imports) not in sys.path:
    sys.path.insert(0, str(project_root_for_imports))

from config import get_paths, config_params

# --- Configuration ---
# NEW: Define a list of penalties to test, from least sensitive to most sensitive
PENALTIES_TO_TEST = [2.5, 2.0, 1.5, 1.0, 0.8] 
OUTPUT_FILENAME = "training_analysis_comparative.png"

def plot_supervisor_analysis(loss_log_path, output_filename):
    """
    Loads training loss data, performs changepoint analysis for multiple penalty
    values, and generates a comprehensive comparative plot.
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

    # --- Changepoint Detection for multiple penalties---
    self_play_data = df[df['game_type'] == 'self-play']['policy_loss_ma'].dropna()
    
    all_changepoints = {}
    if len(self_play_data) > 1:
        print("Pre-calculating changepoint algorithm...")
        algo = rpt.Pelt(model="rbf").fit(self_play_data.values)
        
        for penalty in PENALTIES_TO_TEST:
            try:
                result = algo.predict(pen=penalty)
                valid_indices = [idx for idx in result[:-1] if idx < len(self_play_data)]
                changepoint_indices = self_play_data.index[valid_indices]
                changepoints = df['game'].iloc[changepoint_indices].tolist()
                all_changepoints[penalty] = changepoints
                print(f"  - Penalty={penalty}: Found {len(changepoints)} changepoints.")
            except Exception as e:
                print(f"Could not predict changepoints for penalty {penalty}: {e}")
                all_changepoints[penalty] = []

    # --- Plotting ---
    print("Generating comparative plot...")
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Plot Policy Loss
    ax1.plot(df['game'], df['policy_loss_ma'], color='deepskyblue', label=f'Policy Loss ({rolling_window}-game MA)', zorder=5, linewidth=2.5)
    ax1.set_xlabel('Game Number', fontsize=14)
    ax1.set_ylabel('Policy Loss (Smoothed)', color='deepskyblue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='deepskyblue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add shaded regions for training modes
    last_game = df['game'].max()
    for i in range(len(df) - 1):
        start_game, end_game = df['game'].iloc[i], df['game'].iloc[i+1]
        game_type = df['game_type'].iloc[i]
        color = 'lightcyan' if game_type == 'self-play' else 'lightgreen'
        ax1.axvspan(start_game, end_game, facecolor=color, alpha=0.3, zorder=0)
    if len(df) > 0:
        last_game_type = df['game_type'].iloc[-1]
        last_color = 'lightcyan' if last_game_type == 'self-play' else 'lightgreen'
        ax1.axvspan(df['game'].iloc[-1], last_game, facecolor=last_color, alpha=0.3, zorder=0)

    # Add vertical lines for each set of detected changepoints
    colors = ['darkviolet', 'red', 'orange', 'green', 'magenta']
    linestyles = [':', '--', '-.', '-', ':']
    
    legend_handles = []

    for i, penalty in enumerate(PENALTIES_TO_TEST):
        for cp in all_changepoints.get(penalty, []):
            ax1.axvline(x=cp, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=2, zorder=10)
        # Create a proxy artist for the legend
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], lw=2, label=f'Changepoints (Penalty={penalty})'))


    # --- Final Touches ---
    fig.suptitle('Comparative Supervisor Analysis: Penalty Sensitivity', fontsize=20, weight='bold')
    ax1.set_title(f'Policy Loss vs. Game Number (up to game {int(last_game)})', fontsize=16)
    
    from matplotlib.patches import Patch
    lines, labels = ax1.get_legend_handles_labels()
    
    # Combine all legend elements
    final_legend_elements = lines + legend_handles + [
        Patch(facecolor='lightcyan', alpha=0.5, label='Self-Play Mode'),
        Patch(facecolor='lightgreen', alpha=0.5, label='Mentor Mode')
    ]
    
    fig.legend(handles=final_legend_elements, loc='upper right', bbox_to_anchor=(0.92, 0.88))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, dpi=300)
    print(f"\nComparative analysis plot saved to: {output_filename}")
    plt.close()


if __name__ == '__main__':
    print("Initializing paths via config.get_paths()...")
    paths = get_paths()
    
    data_root = paths.drive_project_root
    log_file_path = data_root / 'loss_log_v2.csv'
    
    print(f"Attempting to load log file from persistent storage: {log_file_path}")
    
    if not log_file_path.exists():
        print(f"Error: Could not find 'loss_log_v2.csv' in '{data_root}'")
        print("Please ensure 'run_training.py' has been run and has generated a loss log file.")
    else:
        # Generate the plot in the local project root for easy access
        output_path = paths.local_project_root / OUTPUT_FILENAME
        plot_supervisor_analysis(log_file_path, output_path)