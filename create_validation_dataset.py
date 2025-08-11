import torch
import chess.pgn
from pathlib import Path
import sys
from tqdm import tqdm
from collections import deque
import gc

# --- Import from project files ---
sys.path.append(str(Path(__file__).resolve().parent))
from config import get_paths
from hardware_setup import get_device
from train_iterative_supervised import process_pgn_stream_to_samples, SAMPLES_PER_BATCH

def main():
    print("--- Creating Pre-processed & Chunked Validation Dataset ---")
    paths = get_paths()
    device = get_device()
    
    validation_corpus_dir = paths.drive_project_root / 'validation_corpus'
    output_dir = paths.drive_project_root / 'training_data' / 'validation_chunks'
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_pgn_files = list(validation_corpus_dir.glob('*.pgn'))
    if not validation_pgn_files:
        print(f"[FATAL] No validation PGN files found in {validation_corpus_dir}")
        sys.exit(1)
        
    chunk_num = 0
    for pgn_path in tqdm(validation_pgn_files, desc="Processing validation PGNs"):
        with open(pgn_path, encoding='utf-8', errors='ignore') as pgn_stream:
            sample_generator = process_pgn_stream_to_samples(pgn_stream, device, None)
            is_file_done = False
            while not is_file_done:
                samples = []
                for _ in range(SAMPLES_PER_BATCH):
                    sample = next(sample_generator, None)
                    if sample is None:
                        is_file_done = True
                        break
                    samples.append(sample)
                
                if samples:
                    chunk_num += 1
                    output_path = output_dir / f"validation_dataset_part_{chunk_num}.pt"
                    print(f"\nSaving validation chunk {chunk_num} with {len(samples)} samples to {output_path}")
                    torch.save(samples, output_path)
                    del samples
                    gc.collect()

    print("\n✅ Chunked validation dataset created successfully.")

if __name__ == "__main__":
    main()