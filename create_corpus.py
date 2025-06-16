#
# File: create_corpus.py
#
"""
A master script to automate the generation of an analysis corpus.

This script finds the latest model checkpoint and the most recent N games,
runs the export_game_analysis.py script for each one, and then automatically
creates a new, executable shell script to assemble all the generated frames
into GIFs using ImageMagick with resource limits to prevent crashes.
"""
import os
import re
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def find_latest_checkpoint(checkpoint_dir):
    """Finds the checkpoint file with the highest game number."""
    path = Path(checkpoint_dir)
    checkpoints = list(path.glob('checkpoint_game_*.pth.tar'))
    
    if not checkpoints:
        return None
        
    latest_file = max(checkpoints, key=lambda p: int(re.search(r'game_(\d+)_', str(p)).group(1)))
    return latest_file

def find_recent_pgns(pgn_dir, num_games):
    """Finds the N most recent PGN files based on game number."""
    path = Path(pgn_dir)
    pgn_files = list(path.glob('*.pgn'))
    
    game_files = []
    for pgn_file in pgn_files:
        match = re.search(r'game_(\d+)\.pgn', pgn_file.name)
        if match:
            game_num = int(match.group(1))
            game_files.append((game_num, pgn_file))
            
    game_files.sort(key=lambda x: x[0], reverse=True)
    
    return [p[1] for p in game_files[:num_games]]

def main():
    parser = argparse.ArgumentParser(description="Automate the creation of an analysis corpus.")
    parser.add_argument("--num-games", type=int, default=50, help="Number of recent games to process.")
    parser.add_argument("--pgn-dir", type=str, default="pgn_games", help="Directory containing PGN files.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory containing model checkpoints.")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Directory to save analysis artifacts.")
    parser.add_argument("--no-loop-gif", action="store_true", help="Generate GIFs that play only once.")
    args = parser.parse_args()

    print("--- Starting Analysis Corpus Generation ---")

    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if not latest_checkpoint:
        print(f"Error: No checkpoints found in '{args.checkpoint_dir}'.")
        return
    print(f"Found latest model: {latest_checkpoint.name}")

    recent_pgns = find_recent_pgns(args.pgn_dir, args.num_games)
    if not recent_pgns:
        print(f"Error: No PGN files with game numbers found in '{args.pgn_dir}'.")
        return
    print(f"Found {len(recent_pgns)} recent games to process (up to {args.num_games}).")

    print("\n--- Processing Games ---")
    all_convert_commands = []
    for i, pgn_path in enumerate(recent_pgns):
        print(f"\nProcessing game {i+1}/{len(recent_pgns)}: {pgn_path.name}...")
        
        command = [
            "python",
            "visualization/export_game_analysis.py",
            "--model_path", str(latest_checkpoint),
            "--pgn_path", str(pgn_path),
            "--output_dir", args.output_dir
        ]
        if args.no_loop_gif:
            command.append("--no-loop")
            
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
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

    print("\n\n--- Corpus Generation Complete ---")
    
    if not all_convert_commands:
        print("⚠️ No 'convert' commands were generated. Nothing to do.")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    script_name = f"generate_gifs_{timestamp}.sh"
    
    with open(script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# This script was generated automatically by create_corpus.py\n\n")
        for cmd in all_convert_commands:
            # Add robust resource limits to prevent ImageMagick from crashing
            parts = cmd.split(' ', 1)
            command_with_limits = f"{parts[0]} -limit memory 1GiB -limit map 2GiB -define registry:temporary-path=/tmp {parts[1]}"
            f.write(command_with_limits + "\n")
            
    os.chmod(script_name, 0o775)
    
    print("✅ All PNG frames and log files have been generated.")
    print(f"✅ A new, executable shell script has been created: {script_name}")
    print("\nTo create all the final GIFs, simply run the new script in your terminal:")
    print("---")
    print(f"./{script_name}")
    print("---")


if __name__ == "__main__":
    main()
