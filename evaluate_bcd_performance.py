import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
import time

# --- Configuration ---
# --- FIX: Point to the new standard log file name ---
CSV_PATH = 'loss_log.csv'
TARGET_COLUMN = 'value_loss'
# A smaller penalty value is more sensitive and better for finding subtle shifts,
# which is ideal for our value_loss, as it's mostly zero.
PENALTY_VALUE = 0.001 
FIGURE_PATH = 'bcd_analysis_plot.png'

def analyze_changepoints(csv_path, column_name, pen, figure_path):
    """
    Performs and visualizes Bayesian Changepoint Detection on training data.
    """
    print(f"--- Starting BCD Analysis on '{column_name}' from '{csv_path}' ---")
    start_time = time.time()

    # 1. Load the aggregated data
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path}. Found {len(df)} game records.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        print("This file will be generated automatically when you run run_training.py.")
        return

    # Group by game number and average losses, in case of multiple epochs per game
    df = df.groupby('game').mean().reset_index()

    points = df[column_name].values

    # 2. Perform changepoint detection using the Pelt algorithm
    print(f"Performing changepoint detection with penalty={pen}...")
    algo = rpt.Pelt(model="l2").fit(points)
    result = algo.predict(pen=pen)
    
    duration = time.time() - start_time
    print(f"Detection complete in {duration:.4f} seconds.")
    # The result includes the end of the series, so len(result)-1 is the number of changepoints.
    print(f"Detected {len(result) - 1} changepoint(s). Indices: {result}")

    # 3. Visualize and save the results
    print(f"Generating and saving plot to {figure_path}...")
    fig, (ax,) = rpt.display(points, result, figsize=(16, 7))
    ax.set_title(f'Changepoint Analysis of {column_name.replace("_", " ").title()}', fontsize=16)
    ax.set_xlabel('Game Number', fontsize=12)
    ax.set_ylabel(column_name.replace("_", " ").title(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close(fig)
    print("--- Analysis Finished ---")

if __name__ == '__main__':
    analyze_changepoints(CSV_PATH, TARGET_COLUMN, PENALTY_VALUE, FIGURE_PATH)