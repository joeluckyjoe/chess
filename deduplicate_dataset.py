import torch
from pathlib import Path
import sys
from tqdm import tqdm
import pickle

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))
from config import get_paths

def main():
    """
    Loads all dataset chunks, de-duplicates them in a memory-efficient way,
    and saves a final, clean dataset.
    """
    print("--- Starting Dataset De-duplication (Memory-Efficient) ---")
    paths = get_paths()
    data_dir = paths.drive_project_root / 'training_data'
    
    chunk_files = sorted(list(data_dir.glob("supervised_dataset_part_*.pt")))
    if not chunk_files:
        print(f"[FATAL] No dataset chunks found in {data_dir}")
        sys.exit(1)
        
    print(f"Found {len(chunk_files)} dataset chunks to process.")

    unique_samples_hashes = set()
    
    # Save the final, clean dataset
    final_save_path = data_dir / "supervised_dataset_final_deduplicated.pkl"
    
    total_unique_count = 0

    # Open the output file for writing in binary append mode
    with open(final_save_path, 'wb') as f_out:
        for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
            data_chunk = torch.load(chunk_file)
            for sample in tqdm(data_chunk, desc=f"Scanning {chunk_file.name}", leave=False):
                # The last CNN tensor in the sequence represents the current board state.
                last_cnn_tensor = sample['state_sequence'][-1][1]
                # Using a hash of the tensor's data is more memory-efficient for the set
                sample_key = hash(last_cnn_tensor.storage().tobytes())

                if sample_key not in unique_samples_hashes:
                    unique_samples_hashes.add(sample_key)
                    # Write the unique sample to the file immediately
                    pickle.dump(sample, f_out)
                    total_unique_count += 1

    print(f"\nDe-duplication complete.")
    print(f"Total unique samples found and saved: {total_unique_count}")
    print(f"Final dataset saved to: {final_save_path}")

if __name__ == "__main__":
    main()