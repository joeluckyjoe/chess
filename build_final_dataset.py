import pickle
from pathlib import Path
import sys
from tqdm import tqdm
import torch
import os

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))
from config import get_paths

# --- Configuration ---
PROGRESS_FILENAME = "build_progress.pkl"
FINAL_DATASET_FILENAME = "supervised_dataset_final.pt"

def save_progress(output_dir, last_chunk_processed):
    """Saves the name of the last chunk that was successfully processed."""
    progress_path = output_dir / PROGRESS_FILENAME
    with open(progress_path, 'wb') as f:
        pickle.dump({'last_chunk_processed': last_chunk_processed}, f)

def load_progress(output_dir):
    """Loads the name of the last chunk that was successfully processed."""
    progress_path = output_dir / PROGRESS_FILENAME
    if progress_path.exists():
        print(f"Found existing progress file. Loading...")
        with open(progress_path, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    """
    Stage 3: Build Final Dataset.
    Loads the set of unique hashes, then iterates through the original data chunks
    one last time to find and save only the unique samples to a final file.
    """
    print("--- Stage 3: Starting Final Dataset Construction (Restartable) ---")
    paths = get_paths()
    data_dir = paths.drive_project_root / 'training_data'
    
    # --- Load the set of unique hashes ---
    hashes_path = data_dir / "unique_hashes.pkl"
    if not hashes_path.exists():
        print(f"[FATAL] unique_hashes.pkl not found at: {hashes_path}")
        sys.exit(1)
        
    print("Loading the set of unique hashes...")
    with open(hashes_path, 'rb') as f:
        unique_hashes = pickle.load(f)
    print(f"Successfully loaded {len(unique_hashes)} unique hashes.")

    # --- Find original data chunks ---
    chunk_files = sorted(list(data_dir.glob("supervised_dataset_part_*.pt")))
    if not chunk_files:
        print(f"[FATAL] No dataset chunks found in {data_dir}")
        sys.exit(1)

    # --- Resume Logic ---
    last_processed_chunk_index = -1
    progress = load_progress(data_dir)
    if progress:
        last_file = progress['last_chunk_processed']
        try:
            last_processed_chunk_index = [str(f) for f in chunk_files].index(str(last_file))
            print(f"Resuming after: {Path(last_file).name}")
        except ValueError:
            print("Warning: Last processed file not found. Starting from beginning.")
    
    # --- Build the final dataset ---
    final_dataset_path = data_dir / FINAL_DATASET_FILENAME
    write_mode = 'ab' if last_processed_chunk_index > -1 else 'wb'
    
    found_unique_samples = 0
    
    with open(final_dataset_path, write_mode) as f_out:
        # Start the loop from the next file to be processed
        for i in tqdm(range(last_processed_chunk_index + 1, len(chunk_files)), desc="Processing Chunks"):
            chunk_file = chunk_files[i]
            try:
                data_chunk = torch.load(chunk_file, map_location=torch.device('cpu'))
                for sample in data_chunk:
                    last_cnn_tensor = sample['state_sequence'][-1][1]
                    sample_key = hash(last_cnn_tensor.cpu().numpy().tobytes())
                    
                    if sample_key in unique_hashes:
                        pickle.dump(sample, f_out)
                        found_unique_samples += 1
                        unique_hashes.remove(sample_key)
                
                # After successfully processing a chunk, save progress
                save_progress(data_dir, chunk_file)
                        
            except Exception as e:
                print(f"Error processing {chunk_file.name}: {e}")
                continue

    print("\n--- Final Dataset Construction Complete ---")
    
    # Final verification of sample count
    with open(final_dataset_path, 'rb') as f:
        final_count = 0
        while True:
            try:
                pickle.load(f)
                final_count += 1
            except EOFError:
                break
    
    print(f"Total unique samples written to final file: {final_count}")
    print(f"Final dataset saved to: {final_dataset_path}")

if __name__ == "__main__":
    main()