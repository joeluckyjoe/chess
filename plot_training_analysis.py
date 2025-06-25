import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import ruptures as rpt
from pathlib import Path

# --- Configuration ---
# You can adjust this penalty value to tune the sensitivity of the changepoint detection.
CHANGEPOINT_PENALTY = 2 
LOG_FILE = 'loss_log_v2.csv'
OUTPUT_FILENAME = "supervisor_analysis_plot.png"

def plot_supervisor_analysis(loss_log_path, output_filename):
    """
    Loads training loss data, performs changepoint analysis, and generates a comprehensive plot.
    """
    try:
        df = pd.read_csv(loss_log_path)
        print(f"Successfully loaded loss data from '{loss_log_path}' with {len(df)} entries.")
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
            changepoint_indices = self_play_data.index[result[:-1]]
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
    for i in range(len(df) - 1):
        start_game, end_game = df['game'].iloc[i], df['game'].iloc[i+1]
        game_type = df['game_type'].iloc[i]
        color = 'lightcyan' if game_type == 'self-play' else 'lightgreen'
        ax1.axvspan(start_game, end_game, facecolor=color, alpha=0.3, zorder=0)

    # Fill the last segment
    if len(df) > 0:
        ax1.axvspan(df['game'].iloc[-1], last_game, facecolor=color, alpha=0.3, zorder=0)

    # Add vertical lines for detected changepoints
    for cp in changepoints:
        ax1.axvline(x=cp, color='darkviolet', linestyle=':', linewidth=2, label=f'Detected Changepoint', zorder=10)
    
    # --- Final Touches ---
    fig.suptitle('Supervisor Training Analysis', fontsize=20, weight='bold')
    ax1.set_title(f'Policy/Value Loss vs. Game Number (up to game {last_game})', fontsize=16)
    
    # Create a single, clean legend
    from matplotlib.patches import Patch
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    legend_elements = lines + lines2 + [
        Patch(facecolor='lightcyan', alpha=0.5, label='Self-Play Mode'),
        Patch(facecolor='lightgreen', alpha=0.5, label='Mentor Mode')
    ]
    
    unique_labels = {}
    for line, label in zip(legend_elements, [l.get_label() for l in legend_elements]):
        if label not in unique_labels:
            unique_labels[label] = line
            
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(0.9, 0.88))
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    plt.savefig(output_filename, dpi=300)
    print(f"\nAnalysis plot saved to: {output_filename}")
    plt.close()


if __name__ == '__main__':
    if not Path(LOG_FILE).exists():
        print(f"Could not find {LOG_FILE}. Please download it from your Google Drive and place it in the project root.")
    else:
        plot_supervisor_analysis(LOG_FILE, OUTPUT_FILENAME)
