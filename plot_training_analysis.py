import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_curves(log_file='loss_log_v2.csv', last_n_games=None):
    """
    Reads the training log file and plots the policy and value loss curves.

    Args:
        log_file (str): The path to the loss_log_v2.csv file.
        last_n_games (int, optional): If specified, plots only the last N games. Defaults to None.
    """
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"Error: The log file '{log_file}' was not found.")
        print("Please make sure you have downloaded it from Google Drive and placed it in the project root.")
        return

    if 'policy_loss' not in df.columns:
        print("Error: 'policy_loss' column not found in the log file.")
        return

    # --- Filter for the last N games as requested ---
    if last_n_games is not None and isinstance(last_n_games, int):
        if len(df) >= last_n_games:
            print(f"Log file has {len(df)} entries. Plotting only the most recent {last_n_games} games.")
            df = df.tail(last_n_games).reset_index(drop=True)
        else:
            print(f"Requested to plot the last {last_n_games} games, but log only contains {len(df)}. Plotting all available data.")

    # Set the style for the plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 7))

    # Create a rolling average to smooth the curve and see the trend
    window_size = 10
    df['policy_loss_smoothed'] = df['policy_loss'].rolling(window=window_size).mean()

    # Plot the raw policy loss
    sns.lineplot(x='game', y='policy_loss', data=df, alpha=0.3, label='Raw Policy Loss')

    # Plot the smoothed policy loss
    sns.lineplot(x='game', y='policy_loss_smoothed', data=df, linewidth=2.5, label=f'Smoothed Policy Loss (window={window_size})')

    # Add a horizontal line for the supervisor's threshold
    supervisor_threshold = 1.8 
    plt.axhline(y=supervisor_threshold, color='r', linestyle='--', linewidth=2, label=f'Supervisor Threshold ({supervisor_threshold})')

    plt.title('Policy Loss During Training (Most Recent Games)', fontsize=16)
    plt.xlabel('Game Number', fontsize=12)
    plt.ylabel('Policy Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Set y-axis limits to better visualize the high loss values
    # We start from 0 to get a proper sense of scale.
    plt.ylim(bottom=0, top=max(df['policy_loss'].max() * 1.1, supervisor_threshold * 1.5))
    
    # --- CHANGE: Save the plot to a file instead of displaying it interactively ---
    output_filename = 'policy_loss_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {output_filename}")
    # plt.show() # This requires a GUI and will not work in all environments.


if __name__ == '__main__':
    # We now call the function telling it to plot only the last 180 games.
    plot_loss_curves(last_n_games=180)
