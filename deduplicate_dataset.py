import torch
from pathlib import Path
import sys
from tqdm import tqdm

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))
from config import get_paths

def main():
    """
    Loads all dataset chunks, de-duplicates them based on the board state,
    and saves a final, clean dataset.
    """
    print("--- Starting Dataset De-duplication ---")
    paths = get_paths()
    data_dir = paths.drive_project_root / 'training_data'
    
    chunk_files = sorted(list(data_dir.glob("supervised_dataset_part_*.pt")))
    if not chunk_files:
        print(f"[FATAL] No dataset chunks found in {data_dir}")
        sys.exit(1)
        
    print(f"Found {len(chunk_files)} dataset chunks to process.")

    unique_samples = set()
    deduplicated_data = []

    for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
        data_chunk = torch.load(chunk_file)
        for sample in tqdm(data_chunk, desc=f"Scanning {chunk_file.name}", leave=False):
            # We create a unique key for each board state.
            # The last CNN tensor in the sequence represents the current board state.
            # We convert its data to a tuple to make it hashable for our set.
            last_cnn_tensor = sample['state_sequence'][-1][1]
            sample_key = tuple(last_cnn_tensor.flatten().tolist())

            if sample_key not in unique_samples:
                unique_samples.add(sample_key)
                deduplicated_data.append(sample)

    print(f"\nDe-duplication complete.")
    print(f"Total unique samples found: {len(deduplicated_data)}")

    # Save the final, clean dataset
    final_save_path = data_dir / "supervised_dataset_final_deduplicated.pt"
    print(f"Saving final dataset to: {final_save_path}")
    torch.save(deduplicated_data, final_save_path)
    print("✅ Final dataset saved successfully.")

if __name__ == "__main__":
    main()