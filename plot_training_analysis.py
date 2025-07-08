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
OUTPUT_FILENAME = "training_progress.png"
ROLLING_WINDOW = 10 # Use a 10-game rolling average to smooth the loss curve

def plot_training_progress(loss_log_path, output_filename):
    """
    Loads training loss data and generates a plot showing the policy loss trend.
    """
    try:
        df = pd.read_csv(loss_log_path)
        print(f"Successfully loaded loss data from '{loss_log_path.name}' with {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: Loss log file not found at '{loss_log_path}'")
        return

    # --- Data Preparation ---
    if 'game' not in df.columns or 'policy_loss' not in df.columns:
        print("Error: The log file is missing required 'game' or 'policy_loss' columns.")
        return
        
    df = df.sort_values('game').reset_index(drop=True)
    df['policy_loss_ma'] = df['policy_loss'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    mentor_games = df[df['game_type'] == 'mentor-play']

    # --- Plotting ---
    print("Generating training progress plot...")
    fig, ax = plt.subplots(figsize=(18, 9))

    # Plot the smoothed policy loss
    ax.plot(df['game'], df['policy_loss_ma'], color='deepskyblue', label=f'Policy Loss ({ROLLING_WINDOW}-game MA)', zorder=5, linewidth=2)
    
    # Scatter plot the raw policy loss points for more detail
    ax.scatter(df['game'], df['policy_loss'], color='deepskyblue', alpha=0.3, s=10, label='Raw Policy Loss', zorder=4)

    # Add vertical lines for mentor games
    for game_num in mentor_games['game']:
        ax.axvline(x=game_num, color='red', linestyle='--', linewidth=1.5, label='Mentor Game Intervention' if 'Mentor Game Intervention' not in [h.get_label() for h in ax.get_legend_handles_labels()[0]] else "")

    ax.set_xlabel('Game Number', fontsize=14)
    ax.set_ylabel('Policy Loss', fontsize=14)
    ax.tick_params(axis='y', labelcolor='deepskyblue')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Final Touches ---
    last_game = df['game'].max()
    fig.suptitle('Agent Training Progress', fontsize=20, weight='bold')
    ax.set_title(f'Policy Loss vs. Game Number (up to game {int(last_game)})', fontsize=16)
    
    # Create the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, dpi=300)
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