import torch
from pathlib import Path
import sys
from tqdm import tqdm
import pickle
import re
import os

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))
from config import get_paths

# --- Configuration ---
PROGRESS_FILENAME = "deduplication_progress.pkl"
FINAL_DATASET_FILENAME = "supervised_dataset_final_deduplicated.pkl"
CLEAR_CACHE_INTERVAL = 50 # Clear cache every 50 chunks

def save_progress(output_dir, last_chunk_processed, hashes_set):
    progress_path = output_dir / PROGRESS_FILENAME
    with open(progress_path, 'wb') as f:
        pickle.dump({'last_chunk_processed': last_chunk_processed, 'hashes': hashes_set}, f)

def load_progress(output_dir):
    progress_path = output_dir / PROGRESS_FILENAME
    if progress_path.exists():
        print(f"Found existing progress file. Loading...")
        with open(progress_path, 'rb') as f:
            return pickle.load(f)
    return None

def clear_system_cache():
    """Runs apt-get clean to clear the local runtime's cache."""
    print("\n--- Clearing system cache to free up disk space... ---")
    os.system('apt-get clean')
    print("--- Cache cleared. ---")

def main():
    print("--- Starting Dataset De-duplication (Resumable & Self-Cleaning) ---")
    paths = get_paths()
    data_dir = paths.drive_project_root / 'training_data'
    
    chunk_files = sorted(list(data_dir.glob("supervised_dataset_part_*.pt")))
    if not chunk_files:
        print(f"[FATAL] No dataset chunks found in {data_dir}")
        sys.exit(1)
        
    print(f"Found {len(chunk_files)} dataset chunks to process.")

    unique_samples_hashes = set()
    last_processed_chunk_index = -1
    progress = load_progress(data_dir)
    if progress:
        unique_samples_hashes = progress['hashes']
        last_file = progress['last_chunk_processed']
        try:
            last_processed_chunk_index = [str(f) for f in chunk_files].index(str(last_file))
            print(f"Resuming after: {Path(last_file).name}")
        except ValueError:
            print("Warning: Last processed file not found. Starting from beginning.")

    final_save_path = data_dir / FINAL_DATASET_FILENAME
    write_mode = 'ab' if last_processed_chunk_index > -1 else 'wb'
    total_unique_count = len(unique_samples_hashes)
    chunks_processed_since_cache_clear = 0

    with open(final_save_path, write_mode) as f_out:
        for i in tqdm(range(last_processed_chunk_index + 1, len(chunk_files)), desc="Processing chunks"):
            chunk_file = chunk_files[i]
            data_chunk = torch.load(chunk_file)
            
            new_samples_in_chunk = 0
            for sample in tqdm(data_chunk, desc=f"Scanning {chunk_file.name}", leave=False):
                last_cnn_tensor = sample['state_sequence'][-1][1]
                sample_key = hash(last_cnn_tensor.cpu().numpy().tobytes())

                if sample_key not in unique_samples_hashes:
                    unique_samples_hashes.add(sample_key)
                    pickle.dump(sample, f_out)
                    new_samples_in_chunk += 1
            
            total_unique_count += new_samples_in_chunk
            print(f"Found {new_samples_in_chunk} new unique samples in {chunk_file.name}.")
            save_progress(data_dir, chunk_file, unique_samples_hashes)

            # <<< MODIFIED: Periodically clear the system cache >>>
            chunks_processed_since_cache_clear += 1
            if chunks_processed_since_cache_clear >= CLEAR_CACHE_INTERVAL:
                clear_system_cache()
                chunks_processed_since_cache_clear = 0

    print(f"\nDe-duplication complete.")
    print(f"Total unique samples in final dataset: {total_unique_count}")
    print(f"Final dataset saved to: {final_save_path}")

if __name__ == "__main__":
    main()