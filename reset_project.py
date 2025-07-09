import shutil
import sys
from pathlib import Path
from datetime import datetime

# --- Configuration ---
# This script assumes it is run from the project's root directory.
# If that's not the case, you may need to adjust the path logic.
try:
    # Add project root to path to allow importing from config
    sys.path.insert(0, str(Path.cwd()))
    from config import get_paths
except ImportError:
    print("ERROR: Could not import get_paths from config.")
    print("Please ensure you are running this script from the project's root directory,")
    print("and that a valid config.py file exists.")
    sys.exit(1)


def reset_project():
    """
    Resets the project for a new, clean training run.
    1. Archives old logs and PGN files into a timestamped directory.
    2. Deletes old checkpoints, training data, and generated puzzles.
    3. Recreates the necessary empty directories for the new run.
    """
    print("--- Starting Project Reset ---")

    # --- 1. Get Environment-Aware Paths ---
    try:
        paths = get_paths()
        project_root = paths.drive_project_root
        pgn_dir = paths.pgn_games_dir
        checkpoints_dir = paths.checkpoints_dir
        training_data_dir = paths.training_data_dir
        loss_log = project_root / 'loss_log_v2.csv'
        supervisor_log = project_root / 'supervisor_log.txt'
        generated_puzzles_file = project_root / 'generated_puzzles.jsonl'
        print("Successfully loaded paths from config.")
    except Exception as e:
        print(f"ERROR: Failed to get paths from config.py. Error: {e}")
        sys.exit(1)


    # --- 2. Archive Old Logs and Games ---
    print("\n--- Step 1: Archiving old run data... ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = project_root / f"run_archive_{timestamp}"
    archive_dir.mkdir(exist_ok=True)

    files_to_archive = {
        "pgn_games": pgn_dir,
        "loss_log": loss_log,
        "supervisor_log": supervisor_log
    }

    for name, path in files_to_archive.items():
        if path.exists():
            try:
                destination = archive_dir / path.name if path.is_file() else archive_dir / name
                shutil.move(str(path), str(destination))
                print(f"  - Moved {path.name} to archive.")
            except Exception as e:
                print(f"  - WARNING: Could not move {path.name}. Error: {e}")
        else:
            print(f"  - Skipping {path.name}, does not exist.")

    print("Archiving complete.")

    # --- 3. Delete Training Artifacts ---
    print("\n--- Step 2: Deleting old training artifacts... ---")

    dirs_to_delete = {
        "Checkpoints": checkpoints_dir,
        "Training Data": training_data_dir
    }
    files_to_delete = {
        "Generated Puzzles": generated_puzzles_file
    }

    for name, path in dirs_to_delete.items():
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"  - Deleted directory: {name} ({path})")
            except Exception as e:
                print(f"  - WARNING: Could not delete directory {name}. Error: {e}")
        else:
            print(f"  - Skipping directory {name}, does not exist.")

    for name, path in files_to_delete.items():
        if path.exists():
            try:
                path.unlink()
                print(f"  - Deleted file: {name} ({path})")
            except Exception as e:
                print(f"  - WARNING: Could not delete file {name}. Error: {e}")
        else:
            print(f"  - Skipping file {name}, does not exist.")


    # --- 4. Recreate Empty Directories ---
    print("\n--- Step 3: Recreating empty directories for new run... ---")
    dirs_to_recreate = [checkpoints_dir, training_data_dir, pgn_dir]
    for path in dirs_to_recreate:
        try:
            path.mkdir(exist_ok=True, parents=True)
            print(f"  - Ensured directory exists: {path}")
        except Exception as e:
            print(f"  - WARNING: Could not create directory {path}. Error: {e}")


    print("\n✅ Project reset successfully. You are ready to start the new training run.")

if __name__ == "__main__":
    reset_project()
