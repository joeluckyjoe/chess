#
# File: create_corpus.py
#
"""
A master script to automate the generation of a structured analysis corpus.

This script orchestrates the analysis of multiple games by:
1. Finding the most recent PGN files from the data directory.
2. Calling 'visualization/export_game_analysis.py' for each game to generate
   PNG frames and a structured .jsonl analysis file.
3. Consolidating all individual .jsonl files into a single, master
   'analysis_corpus.jsonl' for batch analysis.
4. Generating a shell script to assemble all PNG frames into GIFs.
"""
import os
import re
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path to allow importing from config
project_root_path = Path(os.path.abspath(__file__)).parent
if str(project_root_path) not in sys.path:
    sys.path.insert(0, str(project_root_path))

from config import get_paths

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

def combine_jsonl_logs(output_dir, corpus_path):
    """
    Finds all individual JSON Lines log files and combines them into one corpus file.
    """
    path = Path(output_dir)
    log_files = sorted(list(path.glob('*_analysis.jsonl')))

    if not log_files:
        print("Warning: No individual .jsonl analysis files found to combine.")
        return

    print(f"\n--- Combining {len(log_files)} JSONL Log Files ---")
    with open(corpus_path, 'w', encoding='utf-8') as corpus_file:
        for log_file in log_files:
            print(f"Adding: {log_file.name}")
            corpus_file.write(log_file.read_text(encoding='utf-8'))

    print(f"✅ Successfully created analysis corpus at: {Path(corpus_path).resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Automate the creation of a JSONL analysis corpus.")
    parser.add_argument("--num-games", type=int, default=50, help="Number of recent games to process.")
    parser.add_argument("--no-loop-gif", action="store_true", help="Generate GIFs that play only once.")
    args = parser.parse_args()

    print("--- Starting Analysis Corpus Generation ---")

    paths = get_paths()
    code_project_root = paths.project_root
    pgn_data_dir = paths.pgn_games_dir
    output_dir = paths.analysis_output_dir

    # --- FIX: Clean the output directory before starting ---
    if output_dir.exists():
        print(f"Clearing previous analysis artifacts in: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- END FIX ---

    print(f"Code Project Root: {code_project_root}")
    print(f"PGN Data Source: {pgn_data_dir}")
    print(f"Output Directory: {output_dir}")

    recent_pgns = find_recent_pgns(pgn_data_dir, args.num_games)
    if not recent_pgns:
        print(f"Error: No PGN files with game numbers found in '{pgn_data_dir}'.")
        return
    print(f"Found {len(recent_pgns)} recent games to process (up to {args.num_games}).")

    print("\n--- Processing Games ---")
    all_convert_commands = []

    for i, pgn_path in enumerate(recent_pgns):
        print(f"\nProcessing game {i+1}/{len(recent_pgns)}: {pgn_path.name}...")

        export_script_path = code_project_root / "visualization" / "export_game_analysis.py"

        command = [
            sys.executable,
            str(export_script_path),
            "--pgn_path", str(pgn_path),
            "--output_dir", str(output_dir)
        ]
        if args.no_loop_gif:
            command.append("--no-loop")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            for line in result.stdout.splitlines():
                if "convert -delay" in line:
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

    corpus_path = output_dir / "analysis_corpus.jsonl"
    combine_jsonl_logs(output_dir, corpus_path)

    if not all_convert_commands:
        print("⚠️ No 'convert' commands were generated. Skipping GIF script creation.")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    script_name = code_project_root / f"generate_gifs_{timestamp}.sh"

    with open(script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# This script was generated automatically by create_corpus.py\n\n")
        f.write("# Setting ImageMagick policy to allow larger files (if needed)\n")
        f.write('# sed -i \'s/<policy domain="resource" name="disk" value=".*"\\/>/<policy domain="resource" name="disk" value="8GiB"\\/>/g\' /etc/ImageMagick-6/policy.xml\n\n')

        for cmd in all_convert_commands:
            f.write(cmd + "\n")

    os.chmod(script_name, 0o775)

    print(f"✅ A new, executable shell script has been created: {script_name}")
    print("\nTo create all the final GIFs, simply run the new script in your terminal:")
    print(f"--- \n./{script_name.name}\n---")


if __name__ == "__main__":
    main()
