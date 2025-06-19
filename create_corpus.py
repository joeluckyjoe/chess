#
# File: create_corpus.py
#
"""
A master script to automate the generation of an analysis corpus.

V7: Corrects the combine_logs function to prevent redundant headers,
ensuring a clean, parsable combined_logs.txt file. This is the definitive
fix for the log generation process.
"""
import os
import re
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path to allow importing from config
# Note: We determine the true project_root in main() for robustness.
project_root_for_imports = Path(os.path.abspath(__file__)).parent
if str(project_root_for_imports) not in sys.path:
    sys.path.insert(0, str(project_root_for_imports))

from config import get_paths, config_params

def find_recent_pgns(pgn_dir, num_games):
    """Finds the N most recent PGN files based on game number."""
    path = Path(pgn_dir)
    print(f"Searching for PGNs in: {path.resolve()}")
    if not path.exists():
        print(f"Error: PGN directory does not exist at {path.resolve()}")
        return []
        
    pgn_files = list(path.glob('*.pgn'))
    
    game_files = []
    for pgn_file in pgn_files:
        match = re.search(r'game_(\d+)\.pgn', pgn_file.name)
        if match:
            game_num = int(match.group(1))
            game_files.append((game_num, pgn_file))
            
    game_files.sort(key=lambda x: x[0], reverse=True)
    
    return [p[1] for p in game_files[:num_games]]

def combine_logs(output_dir, combined_log_path):
    """
    Finds all individual log files and combines them into one clean file.
    """
    path = Path(output_dir)
    # Use glob to find all log files and sort them to maintain order
    log_files = sorted(list(path.glob('*_log.txt')))

    if not log_files:
        print("Warning: No individual log files found to combine.")
        return

    print(f"\n--- Combining {len(log_files)} Log Files ---")
    # Open with utf-8 encoding to handle characters correctly.
    with open(combined_log_path, 'w', encoding='utf-8') as combined_file:
        for i, log_file in enumerate(log_files):
            print(f"Adding: {log_file.name}")
            
            # Add a separator only BETWEEN files, not before the first one.
            if i > 0:
                # Using two newlines creates a clean visual break.
                combined_file.write("\n\n") 
            
            # CORRECTED: Simply write the content of each individual log file.
            # The individual logs already contain the necessary headers.
            # This avoids creating redundant/corrupting headers.
            combined_file.write(log_file.read_text(encoding='utf-8'))
    
    # Use resolve() to show the full, absolute path
    print(f"✅ Successfully created combined log file at: {Path(combined_log_path).resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Automate the creation of an analysis corpus.")
    parser.add_argument("--num-games", type=int, default=50, help="Number of recent games to process.")
    parser.add_argument("--pgn-dir", type=str, default="pgn_games", help="Directory containing PGN files, relative to project root.")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Directory to save analysis artifacts.")
    parser.add_argument("--no-loop-gif", action="store_true", help="Generate GIFs that play only once.")
    args = parser.parse_args()

    print("--- Starting Analysis Corpus Generation ---")

    # --- Robust Environment-Aware Path Configuration ---
    if 'COLAB_GPU' in os.environ:
        print("Colab environment detected.")
        # In Colab, the project root is explicitly set to the Drive mount point
        project_root = Path('/content/drive/MyDrive/ChessMCTS_RL')
    else:
        print("Local environment detected.")
        # Locally, the project root is the parent of this script's file.
        project_root = Path(__file__).resolve().parent

    # Use the determined project_root to define paths
    checkpoints_dir = project_root / "checkpoints"
    pgn_dir = project_root / args.pgn_dir
    
    if not os.path.isdir(checkpoints_dir) or not any(Path(checkpoints_dir).iterdir()):
        print(f"Error: Checkpoint directory '{checkpoints_dir}' is empty or does not exist.")
        return

    recent_pgns = find_recent_pgns(pgn_dir, args.num_games)
    if not recent_pgns:
        print(f"Error: No PGN files with game numbers found in '{pgn_dir}'.")
        return
    print(f"Found {len(recent_pgns)} recent games to process (up to {args.num_games}).")

    print("\n--- Processing Games ---")
    all_convert_commands = []
    
    # Ensure output directory exists (relative to the execution directory)
    os.makedirs(args.output_dir, exist_ok=True)

    for i, pgn_path in enumerate(recent_pgns):
        print(f"\nProcessing game {i+1}/{len(recent_pgns)}: {pgn_path.name}...")
        
        command = [
            "python",
            "visualization/export_game_analysis.py",
            "--pgn_path", str(pgn_path),
            "--output_dir", args.output_dir
        ]
        if args.no_loop_gif:
            command.append("--no-loop")
            
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            
            for line in result.stdout.splitlines():
                if line.startswith("convert"):
                    all_convert_commands.append(line)
            print(f"Successfully processed {pgn_path.name}.")
            
        except subprocess.CalledProcessError as e:
            print(f"--- ERROR ---")
            print(f"Failed to process {pgn_path.name}.")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            print(f"-------------")

    print("\n\n--- Post-Processing ---")
    
    # 1. Combine all generated log files
    combined_log_path = project_root / "combined_logs.txt"
    combine_logs(args.output_dir, combined_log_path)

    # 2. Generate the GIF creation script
    if not all_convert_commands:
        print("⚠️ No 'convert' commands were generated. Skipping GIF script creation.")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    script_name = project_root / f"generate_gifs_{timestamp}.sh"
    
    with open(script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# This script was generated automatically by create_corpus.py\n\n")
        f.write("# Setting ImageMagick policy to allow larger files\n")
        f.write('sed -i \'s/<policy domain="resource" name="disk" value=".*"\/>/<policy domain="resource" name="disk" value="8GiB"\/>/g\' /etc/ImageMagick-6/policy.xml\n\n')

        for cmd in all_convert_commands:
            f.write(cmd + "\n")
            
    os.chmod(script_name, 0o775)
    
    print(f"✅ A new, executable shell script has been created: {script_name}")
    print("\nTo create all the final GIFs, simply run the new script in your terminal:")
    print(f"--- \n./{script_name}\n---")

if __name__ == "__main__":
    main()
