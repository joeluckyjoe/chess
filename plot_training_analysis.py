#
# File: plot_training_analysis.py (Refactored for Phase BP)
#
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path to allow importing from config
project_root_for_imports = Path(os.path.abspath(__file__)).parent
if str(project_root_for_imports) not in sys.path:
    sys.path.insert(0, str(project_root_for_imports))

from config import get_paths

# --- Configuration ---
OUTPUT_FILENAME = "training_progress.png"
ROLLING_WINDOW = 20 # Use a 20-game rolling average to better see trends in noisy data

def plot_training_progress(loss_log_path, output_filename):
    """
    Loads training loss data and generates a dual-axis plot showing the
    policy, value, and next-state loss trends over time.
    """
    try:
        df = pd.read_csv(loss_log_path)
        print(f"Successfully loaded loss data from '{loss_log_path.name}' with {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: Loss log file not found at '{loss_log_path}'")
        return

    # --- Data Preparation ---
    # MODIFIED: Updated the list of required columns
    required_cols = ['game', 'policy_loss', 'value_loss', 'next_state_loss', 'game_type']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The log file is missing one or more required columns. Expected: {required_cols}")
        return
        
    df = df.sort_values('game').reset_index(drop=True)
    
    # MODIFIED: Calculate rolling means for all three loss types
    df['policy_loss_ma'] = df['policy_loss'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df['value_loss_ma'] = df['value_loss'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df['next_state_loss_ma'] = df['next_state_loss'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    # Identify intervention games (though in mentor-play, all are the same type)
    intervention_games = df[df['game_type'] != 'self-play']

    # --- Plotting ---
    print("Generating training progress plot...")
    fig, ax1 = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#F5F5F5') # A light gray background

    # --- Axis 1: Policy Loss (Left) ---
    color1 = 'dodgerblue'
    ax1.set_xlabel('Game Number', fontsize=16)
    ax1.set_ylabel('Policy Loss', fontsize=16, color=color1)
    ax1.plot(df['game'], df['policy_loss_ma'], color=color1, label=f'Policy Loss ({ROLLING_WINDOW}-game MA)', zorder=5, linewidth=2.5)
    ax1.scatter(df['game'], df['policy_loss'], color=color1, alpha=0.15, s=15, label='Raw Policy Loss')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.7)

    # --- Axis 2: Value Losses (Right) ---
    # MODIFIED: Re-purposed for Value and Next-State Loss
    ax2 = ax1.twinx()
    color2 = 'mediumseagreen'
    color3 = 'darkorange'
    ax2.set_ylabel('Value / Next-State Loss', fontsize=16, color='black')
    
    ax2.plot(df['game'], df['value_loss_ma'], color=color2, label=f'Value Loss ({ROLLING_WINDOW}-game MA)', zorder=5, linewidth=2.5, linestyle='--')
    ax2.plot(df['game'], df['next_state_loss_ma'], color=color3, label=f'Next-State Loss ({ROLLING_WINDOW}-game MA)', zorder=5, linewidth=2.5, linestyle=':')
    
    # Optional: Scatter for raw value losses if needed, but can be noisy
    # ax2.scatter(df['game'], df['value_loss'], color=color2, alpha=0.15, s=15)
    # ax2.scatter(df['game'], df['next_state_loss'], color=color3, alpha=0.15, s=15)
    
    ax2.tick_params(axis='y', labelsize=12)

    # --- Interventions ---
    # This logic is kept for future compatibility if other game_types are used
    if not intervention_games.empty:
        # Check if a label has already been added to avoid duplicates in the legend
        intervention_label_added = any('Intervention' in h.get_label() for h in ax1.get_legend_handles_labels()[0])
        label = 'Intervention Game' if not intervention_label_added else ""
        ax1.axvline(x=intervention_games['game'].iloc[0], color='red', linestyle='--', linewidth=1.5, label=label)
        for game_num in intervention_games['game'].iloc[1:]:
            ax1.axvline(x=game_num, color='red', linestyle='--', linewidth=1.5)

    # --- Final Touches ---
    last_game = df['game'].max()
    fig.suptitle('Agent Training Progress: Mentor-Play Phase (BP)', fontsize=22, weight='bold')
    ax1.set_title(f'Policy & Value Losses vs. Game Number (up to game {int(last_game)})', fontsize=18)
    
    # --- Combined Legend ---
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=12, facecolor='white', framealpha=0.8)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor())
    print(f"\nTraining progress plot saved to: {output_filename}")
    plt.close()


if __name__ == '__main__':
    print("Initializing paths via config.get_paths()...")
    paths = get_paths()
    
    data_root = paths.drive_project_root
    log_file_path = data_root / 'loss_log_v2.csv'
    
    print(f"Attempting to load log file from persistent storage: {log_file_path}")
    
    if not log_file_path.exists():
        print(f"Error: Could not find 'loss_log_v2.csv' in '{data_root}'")
        print("Please ensure a training script has been run and has generated a loss log file.")
    else:
        # Generate the plot in the persistent drive project root for easy access
        output_path = paths.drive_project_root / OUTPUT_FILENAME
        plot_training_progress(log_file_path, output_path)