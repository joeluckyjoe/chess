import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import bayesian_changepoint_detection.online_changepoint_detection as bcpd
import os
import seaborn as sns
from pathlib import Path

# --- Configuration ---
LOG_FILE = 'loss_log_v2.csv'
# CORRECTED: Changed output to JPG for smaller file size
OUTPUT_FILENAME = 'true_bayesian_analysis.jpg' 
EXPECTED_RUN_LENGTH = 50

def analyze_with_true_bayesian(log_file_path):
    """
    Loads the loss log and performs online Bayesian Changepoint Detection,
    then visualizes the results and prints key findings.
    """
    # Step 1: Load your chess loss data
    try:
        df = pd.read_csv(log_file_path)
        print("Successfully loaded the CSV file.")
    except FileNotFoundError:
        print(f"ERROR: The file '{log_file_path}' was not found.")
        print("Please make sure you have the latest 'loss_log_v2.csv' in your project directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return

    # Step 2: Extract the 'policy_loss' column for analysis
    target_column = 'policy_loss'
    if target_column not in df.columns:
        print(f"\nERROR: Could not find the column '{target_column}'.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    # Use a smoothed version to reduce some noise, but keep the series with its index
    smoothed_series = df[target_column].rolling(window=5, min_periods=1).mean().dropna()
    data = smoothed_series.values
    print(f"Analyzing the '{target_column}' column. Data length: {len(data)}")

    # Step 3: Perform Change Point Detection
    hazard_function = partial(bcpd.constant_hazard, EXPECTED_RUN_LENGTH)
    observation_likelihood = bcpd.StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)

    print("Processing data for Online Change Point Detection...")
    R, maxes = bcpd.online_changepoint_detection(data, hazard_function, observation_likelihood)
    print("Processing complete.")

    # --- NEW: Print changepoints directly to the console ---
    print("\n--- Changepoint Analysis Results ---")
    changepoint_games = []
    # A changepoint is where the most likely run length resets.
    # We detect this by seeing when the current run length is shorter than the previous one.
    # CORRECTED: Iterate up to the length of our input 'data' array to prevent an IndexError.
    for i in range(1, len(data)):
        if maxes[i] < maxes[i-1]:
            # The index `i` corresponds to the index in our `smoothed_series`
            original_df_index = smoothed_series.index[i]
            game_number = df.loc[original_df_index, 'game']
            changepoint_games.append(game_number)

    if changepoint_games:
        print(f"True Bayesian model detected potential changepoints at or near game numbers:")
        print([int(g) for g in changepoint_games]) # Clean up the output formatting
    else:
        print("No significant changepoints detected by the True Bayesian model.")
    
    # Step 4: Visualize the Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    sns.set_style("whitegrid")

    # Plot 1: The original policy_loss data
    ax1.plot(df['game'], df[target_column].rolling(window=10).mean(), label='Smoothed Policy Loss (window=10)')
    ax1.set_title('Training Analysis: Policy Loss and Bayesian Changepoint Probability', fontsize=16)
    ax1.set_ylabel('Policy Loss (Smoothed)', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='major', linestyle='--', alpha=0.6)

    # Plot 2: The run length probability distribution
    R_padded = np.pad(R, ((1, 0), (0, 0)), 'constant')
    # CORRECTED: Add a small epsilon to avoid log(0) warnings.
    epsilon = 1e-9
    ax2.imshow(np.log(R_padded + epsilon).T, cmap='gray_r', aspect='auto', origin='lower', extent=[df['game'].min(), df['game'].max(), 0, len(data)])
    ax2.set_title('BCPD Analysis: Log Probability of Run Length', fontsize=14)
    ax2.set_ylabel('Run Length (Games Since Last Change)', fontsize=12)
    ax2.set_xlabel('Game Number', fontsize=12)
    
    fig.tight_layout()

    # Step 5: Save the final plot to a file
    # CORRECTED: Reduced DPI for smaller file size
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"\nSUCCESS! Analysis complete. Graph saved to {OUTPUT_FILENAME}")
    plt.close()


if __name__ == '__main__':
    if not Path(LOG_FILE).exists():
        print(f"Could not find {LOG_FILE}. Please download it from Google Drive.")
    else:
        analyze_with_true_bayesian(LOG_FILE)
