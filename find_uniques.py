import pickle
from pathlib import Path
import sys
from tqdm import tqdm

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))
from config import get_paths

def main():
    """
    Stage 2: Find Unique Hashes.
    Loads all the generated .hashes files, combines them into a single set
    to find all unique hashes, and saves the final set to a file.
    """
    print("--- Stage 2: Starting Unique Hash Identification ---")
    paths = get_paths()
    data_dir = paths.drive_project_root / 'training_data'
    hashes_dir = data_dir / "hashes"

    if not hashes_dir.exists():
        print(f"[FATAL] Hashes directory not found at: {hashes_dir}")
        sys.exit(1)

    hash_files = sorted(list(hashes_dir.glob("*.hashes")))
    if not hash_files:
        print(f"[FATAL] No .hashes files found in {hashes_dir}")
        sys.exit(1)

    print(f"Found {len(hash_files)} hash files to process.")

    unique_hashes = set()
    for hash_file in tqdm(hash_files, desc="Loading hash files"):
        with open(hash_file, 'rb') as f:
            hashes_in_file = pickle.load(f)
            unique_hashes.update(hashes_in_file)
    
    print(f"\nFound a total of {len(unique_hashes)} unique samples.")

    # Save the final set of unique hashes
    final_hashes_path = data_dir / "unique_hashes.pkl"
    print(f"Saving unique hashes to: {final_hashes_path}")
    with open(final_hashes_path, 'wb') as f:
        pickle.dump(unique_hashes, f)

    print("✅ Stage 2: Unique hash identification complete.")

if __name__ == "__main__":
    main()