import pickle
from pathlib import Path
import sys
from tqdm import tqdm
import torch
import os # <<< ADDED

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))
from config import get_paths

# --- Configuration ---
CLEAR_CACHE_INTERVAL = 50 # <<< ADDED: Clear cache every 50 chunks

def clear_system_cache(): # <<< ADDED
    """Runs apt-get clean to clear the local runtime's cache."""
    print("\n--- Clearing system cache to free up disk space... ---")
    os.system('apt-get clean')
    print("--- Cache cleared. ---")

def main():
    """
    Stage 3: Build Final Dataset.
    Loads the set of unique hashes, then iterates through the original data chunks
    one last time to find and save only the unique samples to a final file.
    """
    print("--- Stage 3: Starting Final Dataset Construction ---")
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
    
    # --- Build the final dataset ---
    final_dataset_path = data_dir / "supervised_dataset_final.pt"
    
    found_unique_samples = 0
    chunks_processed_since_cache_clear = 0 # <<< ADDED
    
    with open(final_dataset_path, 'wb') as f_out:
        for chunk_file in tqdm(chunk_files, desc="Processing Chunks"):
            try:
                data_chunk = torch.load(chunk_file, map_location=torch.device('cpu'))
                for sample in data_chunk:
                    last_cnn_tensor = sample['state_sequence'][-1][1]
                    sample_key = hash(last_cnn_tensor.cpu().numpy().tobytes())
                    
                    if sample_key in unique_hashes:
                        pickle.dump(sample, f_out)
                        found_unique_samples += 1
                        unique_hashes.remove(sample_key)
            except Exception as e:
                print(f"Error processing {chunk_file.name}: {e}")
                continue
            
            # <<< ADDED: Periodically clear the system cache >>>
            chunks_processed_since_cache_clear += 1
            if chunks_processed_since_cache_clear >= CLEAR_CACHE_INTERVAL:
                clear_system_cache()
                chunks_processed_since_cache_clear = 0

    print("\n--- Final Dataset Construction Complete ---")
    print(f"Total unique samples found and saved: {found_unique_samples}")
    print(f"Final dataset saved to: {final_dataset_path}")

if __name__ == "__main__":
    main()