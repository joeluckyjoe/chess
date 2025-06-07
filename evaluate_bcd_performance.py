import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import time
import os

def load_training_data(filepath: str) -> pd.DataFrame:
    """
    Loads training log data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame with training data.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame() # Return empty dataframe

    print(f"Loading training data from {filepath}...")
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return df

def analyze_changepoints(data: np.ndarray, model: str = "l2", pen: int = 3) -> list:
    """
    Performs changepoint detection on the provided data series.

    Args:
        data (np.ndarray): The 1D numpy array of data to analyze (e.g., value_loss).
        model (str): The model to use for cost function (e.g., "l2", "rbf").
        pen (int): The penalty value for the detection algorithm.

    Returns:
        list: A list of indices where changepoints are detected.
    """
    print("Performing Bayesian Changepoint Detection...")
    # Using the Pelt search method for its efficiency and accuracy
    algo = rpt.Pelt(model=model).fit(data)
    result = algo.predict(pen=pen)
    print(f"Changepoint detection complete. Found {len(result)-1} changepoint(s).")
    return result

def plot_changepoints(data: pd.Series, changepoints: list, title: str, ylabel: str, output_filename: str):
    """
    Plots the data series and marks the detected changepoints.

    Args:
        data (pd.Series): The data series that was analyzed.
        changepoints (list): The list of changepoint indices.
        title (str): The title for the plot.
        ylabel (str): The label for the y-axis.
        output_filename (str): The filename to save the plot.
    """
    print(f"Generating plot and saving to {output_filename}...")
    plt.figure(figsize=(16, 6))
    plt.plot(data.index, data.values, label=ylabel, alpha=0.8)
    
    # The result includes the end of the series, so we exclude it for plotting
    for cp in changepoints[:-1]:
        plt.axvline(x=cp, color='r', linestyle='--', lw=2, label=f'Changepoint at Game {cp}')

    plt.title(title, fontsize=16)
    plt.xlabel("Game Number", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    # Improve legend handling to avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print("Plot generated successfully.")

def main():
    """
    Main function to run the BCD analysis on training loss data.
    """
    start_time = time.time()
    
    # Configuration
    DATA_FILE = 'training_log_sample.csv'
    TARGET_COLUMN = 'value_loss'
    PLOT_OUTPUT_FILE = 'value_loss_changepoints.png'

    # Load Data
    df = load_training_data(DATA_FILE)
    if df.empty:
        return

    # For this univariate analysis, we focus on the value_loss.
    # We use the game_number from the CSV as the index.
    df.set_index('game_number', inplace=True)
    signal = df[TARGET_COLUMN].values

    # Analyze for Changepoints
    # The penalty `pen` is a hyperparameter. A higher value leads to fewer changepoints.
    # We can tune this later, but 3 is a reasonable starting point.
    changepoints = analyze_changepoints(signal, model="l2", pen=3)

    # Convert detected indices back to game numbers for clarity
    changepoint_games = [df.index[i-1] for i in changepoints if i < len(df.index)]
    print(f"Changepoints detected near games: {changepoint_games}")

    # Plot Results
    plot_changepoints(
        data=df[TARGET_COLUMN],
        changepoints=changepoint_games,
        title=f'Changepoint Analysis of {TARGET_COLUMN.replace("_", " ").title()}',
        ylabel=TARGET_COLUMN.replace("_", " ").title(),
        output_filename=PLOT_OUTPUT_FILE
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n--- BCD Analyzer Performance ---")
    print(f"Execution Time: {execution_time:.4f} seconds")
    print("---------------------------------")
    print(f"\nPhase 15a (V1) is complete. The script has run and produced a plot: {PLOT_OUTPUT_FILE}")

if __name__ == '__main__':
    main()