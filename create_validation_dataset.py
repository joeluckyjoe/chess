import torch
import chess.pgn
from pathlib import Path
import sys
from tqdm import tqdm

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))
from config import get_paths
from hardware_setup import get_device
from train_iterative_supervised import process_pgn_stream_to_samples # Re-use the processing function

def main():
    print("--- Creating Pre-processed Validation Dataset ---")
    paths = get_paths()
    device = get_device()
    
    validation_corpus_dir = paths.drive_project_root / 'validation_corpus'
    output_path = paths.drive_project_root / 'training_data' / 'validation_dataset.pt'

    validation_pgn_files = list(validation_corpus_dir.glob('*.pgn'))
    if not validation_pgn_files:
        print(f"[FATAL] No validation PGN files found in {validation_corpus_dir}")
        sys.exit(1)
        
    all_validation_samples = []
    for pgn_path in tqdm(validation_pgn_files, desc="Processing validation PGNs"):
        with open(pgn_path, encoding='utf-8', errors='ignore') as pgn_stream:
            # Pass pbar as None since we don't need game-level progress here
            all_validation_samples.extend(list(process_pgn_stream_to_samples(pgn_stream, device, None)))

    print(f"\nProcessed {len(all_validation_samples)} total validation samples.")
    print(f"Saving to: {output_path}")
    torch.save(all_validation_samples, output_path)
    print("✅ Validation dataset created successfully.")

if __name__ == "__main__":
    main()