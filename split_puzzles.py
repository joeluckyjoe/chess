#
# File: split_puzzles.py
#
"""
A utility script to split the master tactical puzzle file into a training set
and a held-out evaluation set.

This script reads from 'tactical_puzzles.jsonl', shuffles the puzzles,
and writes them to 'puzzles_train.jsonl' and 'puzzles_eval.jsonl'.
"""
import random
import os

# --- Configuration ---
INPUT_FILE = "tactical_puzzles.jsonl"
TRAIN_FILE = "puzzles_train.jsonl"
EVAL_FILE = "puzzles_eval.jsonl"
TRAIN_SPLIT_RATIO = 0.8

def main():
    """Main function to execute the split."""
    print(f"Starting puzzle split for '{INPUT_FILE}'...")

    # 1. Validate that the input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please ensure you have generated the puzzle file first.")
        return

    # 2. Read all puzzles from the master file
    with open(INPUT_FILE, 'r') as f:
        puzzles = f.readlines()

    total_puzzles = len(puzzles)
    if total_puzzles == 0:
        print("Warning: The input file is empty. No puzzles to split.")
        return
        
    print(f"Found {total_puzzles} total puzzles.")

    # 3. Shuffle the puzzles randomly for an unbiased split
    random.shuffle(puzzles)
    print("Puzzles have been randomly shuffled.")

    # 4. Determine the split point
    split_index = int(total_puzzles * TRAIN_SPLIT_RATIO)
    
    train_puzzles = puzzles[:split_index]
    eval_puzzles = puzzles[split_index:]

    # 5. Write the training set
    with open(TRAIN_FILE, 'w') as f:
        f.writelines(train_puzzles)
    print(f"-> Saved {len(train_puzzles)} puzzles to '{TRAIN_FILE}'")

    # 6. Write the evaluation set
    with open(EVAL_FILE, 'w') as f:
        f.writelines(eval_puzzles)
    print(f"-> Saved {len(eval_puzzles)} puzzles to '{EVAL_FILE}'")

    print("\n✅ Dataset splitting complete.")

if __name__ == "__main__":
    main()