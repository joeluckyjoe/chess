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
    Stage 1: Generate Hashes.
    Reads each data chunk, computes a hash for every sample, and saves
    these hashes to a corresponding .hashes file. This process is fully resumable.
    """
    print("--- Stage 1: Starting Hash Generation ---")
    paths = get_paths()
    data_dir = paths.drive_project_root / 'training_data'
    
    hashes_dir = data_dir / "hashes"
    hashes_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_files = sorted(list(data_dir.glob("supervised_dataset_part_*.pt")))
    if not chunk_files:
        print(f"[FATAL] No dataset chunks found in {data_dir}")
        sys.exit(1)
        
    print(f"Found {len(chunk_files)} dataset chunks to process.")

    for chunk_file in tqdm(chunk_files, desc="Processing Chunks"):
        output_hash_path = hashes_dir / f"{chunk_file.stem}.hashes"

        if output_hash_path.exists():
            continue

        chunk_hashes = []
        try:
            # <<< MODIFIED: Use torch.load with map_location and iterate through the loaded chunk
            data_chunk = torch.load(chunk_file, map_location=torch.device('cpu'))
            for sample in tqdm(data_chunk, desc=f"Scanning {chunk_file.name}", leave=False, unit="samples"):
                last_cnn_tensor = sample['state_sequence'][-1][1]
                sample_key = hash(last_cnn_tensor.cpu().numpy().tobytes())
                chunk_hashes.append(sample_key)

        except Exception as e:
            print(f"\nError processing {chunk_file.name}: {e}")
            continue
        
        with open(output_hash_path, 'wb') as f_out:
            pickle.dump(chunk_hashes, f_out)

    print("\n✅ Stage 1: Hash generation complete.")

if __name__ == "__main__":
    main()