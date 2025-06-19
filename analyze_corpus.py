#
# File: analyze_corpus.py
#
"""
A dedicated script to parse and analyze the 'combined_logs.txt' file.

This tool is designed to programmatically search for patterns of weakness in
the agent's play, with an initial focus on testing the "checkmate blindness"
hypothesis. This version uses a robust line-by-line parsing method to handle
inconsistencies and potential file encoding issues.

Usage:
    python analyze_corpus.py --log-file path/to/your/combined_logs.txt
"""

import re
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import codecs

def parse_log_file(log_path, debug=False):
    """
    Parses the combined log file to extract game data, focusing on plies
    where a forced mate is present. This version manually reconstructs game
    blocks to be immune to file encoding issues and inconsistent separators.
    
    Args:
        log_path (Path): The path to the combined_logs.txt file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a ply
              with a forced mate and contains relevant information.
    """
    if not log_path.exists():
        print(f"Error: Log file not found at '{log_path}'")
        return []

    try:
        # DEFINITIVE FIX: Use utf-8-sig to automatically handle and strip a 
        # potential Byte Order Mark (BOM) at the start of the file. This was
        # the root cause of the previous parsing failures.
        with open(log_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []

    game_blocks = []
    # Find the indices of all game headers using the robust startswith() method
    start_indices = [i for i, line in enumerate(lines) if line.strip().startswith("Analysis Log for:")]
    
    if not start_indices:
        print("ERROR: No lines starting with 'Analysis Log for:' were found in the log file.")
        return []

    if debug:
        print(f"DEBUG: Found {len(start_indices)} game header lines.")

    # Create game blocks by slicing the file content from one header to the next
    for i in range(len(start_indices)):
        start_line = start_indices[i]
        end_line = start_indices[i+1] if i + 1 < len(start_indices) else len(lines)
        game_blocks.append("".join(lines[start_line:end_line]))
        
    all_mate_plies = []
    
    for game_content in game_blocks:
        game_name_match = re.search(r"Analysis Log for: (.*?)\n", game_content)
        if not game_name_match:
            continue
        game_name = game_name_match.group(1).strip()
        
        # This regex robustly finds all ply blocks within a single game's content
        ply_blocks = re.findall(r"--- Ply (\d+).*?---\n(.*?)(?=\n--- Ply|\Z)", game_content, re.DOTALL)
        
        for ply_num_str, ply_data in ply_blocks:
            ply_num = int(ply_num_str)

            # Use more flexible regex for stockfish eval
            mate_match = re.search(r"Stockfish.+?Mate\(([-]?\d+)\)", ply_data)
            
            if mate_match:
                mate_in_n = int(mate_match.group(1))
                
                # This regex now robustly handles both "Agent Evaluation:" and "Value:" formats
                agent_value = None
                new_format_match = re.search(r"Agent Evaluation: .*Pre-Move=([-]?\d+\.\d+)", ply_data)
                old_format_match = re.search(r"Value: ([-]?\d+\.\d+)", ply_data)
                
                if new_format_match:
                    agent_value = float(new_format_match.group(1))
                elif old_format_match:
                    agent_value = float(old_format_match.group(1))
                
                if agent_value is not None:
                    all_mate_plies.append({
                        'game_name': game_name,
                        'ply': ply_num,
                        'mate_in_n': mate_in_n,
                        'agent_value': agent_value
                    })

    return all_mate_plies

def analyze_checkmate_blindness(mate_data, blindness_threshold=0.95):
    """
    Analyzes the extracted mate data to identify instances of checkmate blindness.
    """
    if not mate_data:
        return pd.DataFrame()
    df = pd.DataFrame(mate_data)
    df['is_blind'] = df['agent_value'].abs() < blindness_threshold
    blind_events = df[df['is_blind']].copy()
    blind_events['mating_side'] = blind_events['mate_in_n'].apply(lambda x: 'White' if x > 0 else 'Black')
    return blind_events

def generate_report(mate_data, blind_events):
    """
    Prints a summary report to the console and generates a visualization.
    """
    print("\n" + "="*50)
    print("      Checkmate Blindness Analysis Report")
    print("="*50 + "\n")
    
    if not mate_data:
        print("No mate positions were found in the log file. Cannot generate a report.")
        return

    total_mate_positions = len(mate_data)
    total_blind_instances = len(blind_events)
    
    print(f"Total forced mate positions found: {total_mate_positions}")
    print(f"Instances of checkmate blindness found: {total_blind_instances}")
    
    if total_mate_positions > 0:
        blindness_rate = (total_blind_instances / total_mate_positions) * 100
        print(f"Blindness Rate: {blindness_rate:.2f}%")
    
    if not blind_events.empty:
        print("\n--- Detailed Blindness Events ---\n")
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.width', 1000)
        print(blind_events.sort_values(by=['game_name', 'ply'])[['game_name', 'ply', 'mate_in_n', 'agent_value', 'mating_side']].to_string(index=False))
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(data=blind_events, x='ply', y='agent_value', hue='mating_side', s=100, ax=ax, palette={'White': '#4a90e2', 'Black': '#d0021b'})
        ax.set_title('Checkmate Blindness: Agent Value vs. Ply Number in Forced Mate Positions', fontsize=16, pad=20)
        ax.set_xlabel('Ply (Move Number)', fontsize=12)
        ax.set_ylabel('Agent\'s Evaluation', fontsize=12)
        ax.axhline(0.95, color='gray', linestyle='--', linewidth=1)
        ax.axhline(-0.95, color='gray', linestyle='--', linewidth=1)
        
        try:
            x_limit = ax.get_xlim()[1]
            ax.text(x_limit, 0.95, '  Confidence Threshold', va='center', ha='left', color='gray')
        except IndexError:
            pass
            
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_filename = "checkmate_blindness_analysis.png"
        plt.savefig(plot_filename)
        print(f"\n✅ Analysis plot saved as '{plot_filename}'")
    else:
        print("\n✅ No instances of checkmate blindness were found based on the current threshold.")
    print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Analyze chess game logs for checkmate blindness.")
    parser.add_argument("--log-file", type=str, default="combined_logs.txt", help="Path to the combined log file.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Confidence threshold below which the agent is considered 'blind' to a mate.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose diagnostic printing for the first game.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    
    print(f"--- Starting Analysis of '{log_path.name}' ---")
    
    mate_data = parse_log_file(log_path, args.debug)
    blind_events = analyze_checkmate_blindness(mate_data, args.threshold)
    generate_report(mate_data, blind_events)

if __name__ == "__main__":
    main()
