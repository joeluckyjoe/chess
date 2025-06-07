import os
import pickle
import pandas as pd
from tqdm import tqdm
import re
import torch # <--- ADD THIS IMPORT

def get_project_root():
    """
    Determines the project root directory based on the execution environment.
    """
    colab_path = '/content/drive/MyDrive/ChessMCTS_RL'
    if os.path.exists('/content/drive'):
        print("Colab environment detected.")
        return colab_path
    else:
        print("Local environment detected.")
        return '.'

# --- Configuration ---
PROJECT_ROOT = get_project_root()
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, 'training_data')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'training_losses.csv')

def get_game_number_from_filename(filename):
    """Extracts the game number from a filename."""
    match = re.search(r'game_(\d+)_data\.pkl', filename)
    if match:
        return int(match.group(1))
    return -1

def aggregate_loss_data(data_dir, output_file):
    """
    Aggregates training loss data from individual game .pkl files into a single CSV.
    """
    loss_records = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        if '/content/drive' in data_dir:
             print("Please check that your Google Drive is mounted and the project path is correct.")
        else:
            print("Please ensure you are running this script from the project's root directory.")
        return

    try:
        filenames = [f for f in os.listdir(data_dir) if f.startswith('game_') and f.endswith('_data.pkl')]
        filenames.sort(key=get_game_number_from_filename)
    except FileNotFoundError:
        print(f"Error: Could not list files in '{data_dir}'.")
        return

    print(f"Found {len(filenames)} game data files in '{data_dir}'. Aggregating...")

    for filename in tqdm(filenames, desc="Processing Game Files"):
        game_num = get_game_number_from_filename(filename)
        if game_num == -1:
            continue

        file_path = os.path.join(data_dir, filename)
        try:
            # --- THIS IS THE FIX ---
            # Use torch.load with map_location to handle GPU/CPU mismatch
            game_data = torch.load(file_path, map_location=torch.device('cpu'))
            # --- END OF FIX ---
            
            if 'policy_loss' in game_data and 'value_loss' in game_data:
                loss_records.append({
                    'game': game_num,
                    'policy_loss': game_data['policy_loss'],
                    'value_loss': game_data['value_loss']
                })
            else:
                print(f"Warning: Loss data not found in {filename}")

        except Exception as e:
            print(f"Warning: Could not process file {filename}. Error: {e}. Skipping.")

    if not loss_records:
        print("No loss records were aggregated. Cannot create CSV.")
        return

    df = pd.DataFrame(loss_records)
    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully aggregated data from {len(df)} games into {output_file}")

if __name__ == '__main__':
    aggregate_loss_data(TRAINING_DATA_DIR, OUTPUT_CSV)