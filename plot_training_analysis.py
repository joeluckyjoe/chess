#
# File: plot_training_analysis.py (Updated for Phase BI)
#
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path to allow importing from config
# This assumes your notebook is in the project root directory.
project_root_for_imports = Path(os.path.abspath(__file__)).parent
if str(project_root_for_imports) not in sys.path:
    sys.path.insert(0, str(project_root_for_imports))

from config import get_paths

# --- Configuration ---
OUTPUT_FILENAME = "training_progress_phase_bi.png"
ROLLING_WINDOW = 10 # Use a 10-game rolling average to smooth the loss curves

def plot_training_progress(loss_log_path, output_filename):
    """
    Loads training loss data and generates a dual-axis plot showing the
    policy loss and material loss trends over time.
    """
    try:
        df = pd.read_csv(loss_log_path)
        print(f"Successfully loaded loss data from '{loss_log_path.name}' with {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: Loss log file not found at '{loss_log_path}'")
        return

    # --- Data Preparation ---
    required_cols = ['game', 'policy_loss', 'material_loss', 'game_type']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The log file is missing one or more required columns: {required_cols}")
        return
        
    df = df.sort_values('game').reset_index(drop=True)
    
    # Calculate rolling means for smoothing
    df['policy_loss_ma'] = df['policy_loss'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df['material_loss_ma'] = df['material_loss'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    # Identify intervention games
    intervention_games = df[df['game_type'] != 'self-play']

    # --- Plotting ---
    print("Generating training progress plot...")
    fig, ax1 = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('whitesmoke')

    # --- Axis 1: Policy Loss (Left) ---
    color1 = 'deepskyblue'
    ax1.set_xlabel('Game Number', fontsize=16)
    ax1.set_ylabel('Policy Loss', fontsize=16, color=color1)
    ax1.plot(df['game'], df['policy_loss_ma'], color=color1, label=f'Policy Loss ({ROLLING_WINDOW}-game MA)', zorder=5, linewidth=2.5)
    ax1.scatter(df['game'], df['policy_loss'], color=color1, alpha=0.2, s=15, label='Raw Policy Loss', zorder=4)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.7)

    # --- Axis 2: Material Loss (Right) ---
    ax2 = ax1.twinx() # Create a second y-axis that shares the same x-axis
    color2 = 'mediumseagreen'
    ax2.set_ylabel('Material Loss', fontsize=16, color=color2)
    ax2.plot(df['game'], df['material_loss_ma'], color=color2, label=f'Material Loss ({ROLLING_WINDOW}-game MA)', zorder=5, linewidth=2.5, linestyle=':')
    ax2.scatter(df['game'], df['material_loss'], color=color2, alpha=0.2, s=15, label='Raw Material Loss', zorder=4)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
    ax2.set_yscale('log') # Use a log scale for material loss due to its wide range

    # --- Interventions ---
    for game_num in intervention_games['game']:
        ax1.axvline(x=game_num, color='red', linestyle='--', linewidth=1.5, label='Mentor Intervention' if 'Mentor Intervention' not in [h.get_label() for h in ax1.get_legend_handles_labels()[0]] else "")

    # --- Final Touches ---
    last_game = df['game'].max()
    fig.suptitle('Agent Training Progress: Phase BI (Value Head Calibration)', fontsize=22, weight='bold')
    ax1.set_title(f'Policy & Material Loss vs. Game Number (up to game {int(last_game)})', fontsize=18)
    
    # --- Combined Legend ---
    # To avoid duplicate labels, we gather handles and labels from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    # Filter out duplicate intervention labels
    unique_handles, unique_labels = [], []
    for handle, label in zip(h1 + h2, l1 + l2):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    ax1.legend(unique_handles, unique_labels, loc='upper right', fontsize=12)
    
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
        print("Please ensure 'run_training.py' has been run and has generated a loss log file.")
    else:
        # Generate the plot in the persistent drive project root for easy access
        output_path = paths.drive_project_root / OUTPUT_FILENAME
        plot_training_progress(log_file_path, output_path)
