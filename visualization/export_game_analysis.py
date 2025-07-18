#
# File: visualization/export_game_analysis.py
#
"""
A script to generate analysis artifacts for completed games, now with dual-mode functionality.

This tool can be run in one of two modes:

1.  analysis (default):
    Generates two primary outputs for a single PGN:
    - Individual PNG frames for creating a visual game analysis GIF.
    - A structured JSON Lines (.jsonl) log file containing detailed,
      machine-readable data for each move, including agent evaluations
      and full symmetric attention data.
    This mode loads the agent's neural network and graphics libraries.

2.  puzzle:
    A high-speed mode for scanning one or more PGN files to find and
    export tactical puzzles based on blunders. It analyzes each move,
    and if a move causes a significant evaluation drop compared to
    Stockfish's best move, it saves the position (FEN) and the correct
    move to a JSONL file. This mode does NOT load the agent network or any
    graphics libraries, making it very fast.
"""
import sys
import os
from datetime import datetime
import json
from pathlib import Path
import argparse

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import chess
import chess.pgn
import chess.engine
import pygame
import numpy as np

# --- Project-specific Imports ---
from config import get_paths, config_params
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.neural_network.gnn_models import SquareGNN, PieceGNN
from gnn_agent.neural_network.attention_module import CrossAttentionModule
from gnn_agent.neural_network.policy_value_heads import PolicyHead, ValueHead
from gnn_agent.gamestate_converters import gnn_data_converter, action_space_converter

# --- Pygame and Board Configuration ---
SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE
INFO_PANE_HEIGHT = 120
TOTAL_HEIGHT = BOARD_SIZE + INFO_PANE_HEIGHT
ASSET_PATH = "assets/pieces"

# Colors
COLOR_LIGHT_SQ = pygame.Color(240, 217, 181)
COLOR_DARK_SQ = pygame.Color(181, 136, 99)
COLOR_INFO_BG = pygame.Color(20, 20, 20)
COLOR_INFO_FONT = pygame.Color(230, 230, 230)
COLOR_MOVING_PIECE_HIGHLIGHT = pygame.Color(255, 255, 0, 150)
COLOR_ATTENTION_1 = pygame.Color(255, 0, 0)
COLOR_ATTENTION_2 = pygame.Color(255, 165, 0)
COLOR_ATTENTION_3 = pygame.Color(255, 255, 0)


def find_latest_checkpoint(checkpoint_dir):
    """Finds the most recently modified checkpoint file in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('*.pth.tar'))
    
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint


def load_model_from_checkpoint(model_path, device):
    """Loads a ChessNetwork model from a .pth checkpoint file."""
    # Note: These parameters should match the saved model's architecture.
    # This is suitable for demonstration but for a robust system, architecture
    # details might be saved in the checkpoint itself.
    square_gnn = SquareGNN(in_features=12, hidden_features=256, out_features=128, heads=4)
    piece_gnn = PieceGNN(in_channels=13, hidden_channels=256, out_channels=128, num_layers=3) # Updated PieceGNN
    attention_module = CrossAttentionModule(sq_embed_dim=128, pc_embed_dim=128, num_heads=4, dropout_rate=0.1)
    policy_head = PolicyHead(embedding_dim=128, num_possible_moves=4672)
    value_head = ValueHead(embedding_dim=128)
    network = ChessNetwork(square_gnn, piece_gnn, attention_module, policy_head, value_head).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    network.load_state_dict(state_dict)
    network.eval()
    print(f"Successfully loaded model from {os.path.basename(str(model_path))}")
    return network


def get_model_outputs_for_board(board, network, device):
    """Gets policy, value, and BOTH symmetric attention tensors."""
    gnn_input_data, piece_labels = gnn_data_converter.convert_to_gnn_input(
        board, device=device, for_visualization=True
    )
    with torch.no_grad():
        policy_logits, value_logit, ps_weights, sp_weights = network(*gnn_input_data, return_attention=True)
    
    value = torch.tanh(value_logit).item()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return [], value, ps_weights, sp_weights, piece_labels, gnn_input_data

    legal_move_indices = [action_space_converter.move_to_index(m, board) for m in legal_moves]
    legal_move_indices_tensor = torch.tensor(legal_move_indices, device=policy_logits.device)

    policy_probs = torch.softmax(policy_logits.flatten(), dim=0)
    # Ensure indices are within the bounds of policy_probs
    legal_move_indices_tensor = legal_move_indices_tensor[legal_move_indices_tensor < len(policy_probs)]
    legal_probs = torch.gather(policy_probs, 0, legal_move_indices_tensor)

    k = min(3, len(legal_moves))
    if k == 0:
        return [], value, ps_weights, sp_weights, piece_labels, gnn_input_data
        
    top_k_probs, top_k_indices_in_legal_list = torch.topk(legal_probs, k)

    top_moves = [(board.san(legal_moves[i]), p.item()) for i, p in zip(top_k_indices_in_legal_list, top_k_probs)]
    return top_moves, value, ps_weights, sp_weights, piece_labels, gnn_input_data


def format_attention_string_for_display(piece_attention_vector, from_square_san, k=3):
    """Formats the top-k attended squares into a string for Pygame display."""
    if piece_attention_vector is None:
        return "Attention: N/A"
    
    num_to_display = min(k, len(piece_attention_vector))
    if num_to_display == 0:
        return f"Attn Focus ({from_square_san}): None"

    top_k_scores, top_k_indices = torch.topk(piece_attention_vector, num_to_display)
    attended_sq_info = [f"{chess.square_name(idx.item())}({s:.2f})" for s, idx in zip(top_k_scores, top_k_indices)]
    return f"Attn Focus ({from_square_san}): {', '.join(attended_sq_info)}"


def get_structured_ps_attention(ps_attention_vector, from_square, k=3):
    """Creates a structured dictionary for the top-k P->S attended squares."""
    if ps_attention_vector is None:
        return None
    
    num_to_display = min(k, len(ps_attention_vector))
    if num_to_display == 0:
        return None
        
    top_k_scores, top_k_indices = torch.topk(ps_attention_vector, num_to_display)
    
    attended_squares = [
        {"square": chess.square_name(idx.item()), "score": round(s.item(), 5)}
        for s, idx in zip(top_k_scores, top_k_indices)
    ]
    
    return {
        "from_square": chess.square_name(from_square),
        "top_k_attended_squares": attended_squares
    }


def get_structured_sp_attention(sp_weights, to_square, piece_labels, k=3):
    """Creates a structured dictionary for the top-k S->P attended pieces."""
    if sp_weights is None or piece_labels is None:
        return None
        
    square_attention_vector = sp_weights[to_square]
    
    num_pieces = square_attention_vector.shape[0]
    if num_pieces == 0:
        return None

    k = min(k, num_pieces)
    top_k_scores, top_k_indices_in_tensor = torch.topk(square_attention_vector, k)
    
    attended_pieces = []
    for score, piece_tensor_idx in zip(top_k_scores, top_k_indices_in_tensor):
        square_of_piece = piece_labels[piece_tensor_idx.item()]
        attended_pieces.append({
            "piece_on_square": chess.square_name(square_of_piece),
            "score": round(score.item(), 5)
        })

    return {
        "to_square": chess.square_name(to_square),
        "top_k_attending_pieces": attended_pieces
    }


def load_piece_images():
    """Loads piece images from the assets directory."""
    images = {}
    for piece_char in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']:
        color = 'w' if piece_char.isupper() else 'b'
        filename = f"{color}{piece_char.upper()}.png"
        path = os.path.join(project_root, ASSET_PATH, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Piece image not found at {path}.")
        img = pygame.image.load(path)
        images[piece_char] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
    return images


def draw_frame(surface, board, piece_images, info_lines, piece_attention_vector=None, moving_piece_square=None, k=3):
    """Draws a single frame, including board, multi-line info pane, and visual attention."""
    # Draw board squares
    for i in range(64):
        row, col = divmod(i, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        color = COLOR_LIGHT_SQ if (row + col) % 2 == 0 else COLOR_DARK_SQ
        surface.fill(color, rect)

    # Draw attention overlay
    if piece_attention_vector is not None:
        gradient_colors = [COLOR_ATTENTION_1, COLOR_ATTENTION_2, COLOR_ATTENTION_3]
        num_to_display = min(k, len(piece_attention_vector))
        if num_to_display > 0:
            top_k_scores, top_k_indices = torch.topk(piece_attention_vector, num_to_display)
            max_score = top_k_scores[0].item() if len(top_k_scores) > 0 else 0

            for i in range(len(top_k_indices)):
                sq_idx = top_k_indices[i].item()
                score = top_k_scores[i].item()
                color = gradient_colors[i]
                alpha = int(100 + (score / max_score) * 155) if max_score > 0 else 100
                row, col = divmod(sq_idx, 8)
                py_row = 7 - row
                rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                overlay.fill((*color[:3], alpha))
                surface.blit(overlay, rect.topleft)

    # Highlight the piece that is about to move
    if moving_piece_square is not None:
        row, col = divmod(moving_piece_square, 8)
        py_row = 7 - row
        rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.circle(surface, COLOR_MOVING_PIECE_HIGHLIGHT, rect.center, SQUARE_SIZE // 2, 5)

    # Draw pieces
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            py_row = 7 - row
            rect = pygame.Rect(col * SQUARE_SIZE, py_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            piece_img = piece_images[piece.symbol()]
            img_rect = piece_img.get_rect(center=rect.center)
            surface.blit(piece_img, img_rect)

    # Draw info pane
    info_rect = pygame.Rect(0, BOARD_SIZE, BOARD_SIZE, INFO_PANE_HEIGHT)
    surface.fill(COLOR_INFO_BG, info_rect)
    font_main = pygame.font.SysFont('monospace', 16)
    y_offset = BOARD_SIZE + 5
    for line in info_lines:
        text_surface = font_main.render(line, True, COLOR_INFO_FONT)
        surface.blit(text_surface, (10, y_offset))
        y_offset += 20


def get_stockfish_eval(engine, board, time_limit=0.1):
    """
    Gets the evaluation from Stockfish.
    Returns a tuple: (formatted_string, analysis_info_dict)
    """
    if engine is None:
        return "N/A", None
    try:
        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info.get("score")
        if not score:
            return "N/A", info

        pov_score = score.pov(board.turn)
        if pov_score.is_mate():
            return f"Mate({pov_score.mate()})", info
        else:
            # Return score from White's perspective for consistency in logs
            return f"{score.white().score()}", info
            
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError) as e:
        print(f"Stockfish engine error: {e}")
        return "Error", None


def run_visual_analysis(args, paths):
    """
    Executes the visual analysis mode, generating frames and a detailed log.
    """
    checkpoints_dir = paths.checkpoints_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = args.model_path
    if not model_path:
        print("Searching for the latest checkpoint...")
        model_path = find_latest_checkpoint(checkpoints_dir)
        if not model_path:
            print(f"Error: No checkpoints found in '{checkpoints_dir}'.")
            return
        print(f"Automatically selected latest model: {os.path.basename(str(model_path))}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pgn_filename = os.path.splitext(os.path.basename(args.pgn_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    frame_dir_name = f"{pgn_filename}_frames_{timestamp}"
    frame_dir_path = output_dir / frame_dir_name
    frame_dir_path.mkdir(exist_ok=True)

    stockfish_engine = None
    stockfish_path_from_config = config_params.get("STOCKFISH_PATH")
    if stockfish_path_from_config and os.path.exists(stockfish_path_from_config):
        try:
            stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path_from_config)
        except Exception as e:
            print(f"Warning: Could not start Stockfish engine: {e}. Continuing without it.")
    else:
        print("Warning: Stockfish path not set or found. Continuing without objective evaluation.")

    try:
        with open(args.pgn_path) as pgn_file:
            game = chess.pgn.read_game(pgn_file)
        if not game:
            print(f"Error: Could not read a valid game from PGN file: {args.pgn_path}")
            return

        network = load_model_from_checkpoint(model_path, device)
        board = game.board()
        pygame.init()
        piece_images = load_piece_images()
        surface = pygame.Surface((BOARD_SIZE, TOTAL_HEIGHT))
        
        frame_count = 0
        json_log_entries = []

        ply = 0
        _, pre_move_value, _, _, _, _ = get_model_outputs_for_board(board, network, device)
        stockfish_eval_str, _ = get_stockfish_eval(stockfish_engine, board)
        
        initial_log_entry = {
            "ply": ply, "move_san": "Initial", "board_fen_before": board.fen(),
            "agent_value_before": round(pre_move_value, 5), "stockfish_eval": stockfish_eval_str
        }
        json_log_entries.append(initial_log_entry)
        
        info_lines = [f"Ply {ply}: Start Position", f"Agent Value: {pre_move_value:.3f}", f"Stockfish Eval: {stockfish_eval_str}"]
        draw_frame(surface, board, piece_images, info_lines)
        pygame.image.save(surface, os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png"))
        
        for move in game.mainline_moves():
            ply += 1
            frame_count +=1
            san_move = board.san(move)
            
            pre_move_fen = board.fen()
            is_capture = board.is_capture(move)
            
            pre_move_policy, pre_move_value, ps_weights, sp_weights, piece_labels, gnn_data = get_model_outputs_for_board(board, network, device)
            stockfish_eval_str, _ = get_stockfish_eval(stockfish_engine, board)
            
            from_square = move.from_square
            to_square = move.to_square

            piece_attention_vector = None
            structured_ps_attention = None
            structured_sp_attention = None

            if ps_weights is not None and gnn_data is not None:
                piece_pos_tensor = gnn_data.piece_to_square_map
                moved_piece_indices = (piece_pos_tensor == from_square).nonzero(as_tuple=True)[0]
                if moved_piece_indices.numel() > 0:
                    piece_idx_in_tensor = moved_piece_indices[0]
                    piece_attention_vector = ps_weights[piece_idx_in_tensor]
                    structured_ps_attention = get_structured_ps_attention(piece_attention_vector, from_square)

            if sp_weights is not None and piece_labels is not None:
                structured_sp_attention = get_structured_sp_attention(sp_weights, to_square, piece_labels)

            policy_str_display = ', '.join([f"{m_san}({p:.2f})" for m_san, p in pre_move_policy])
            attention_str_display = format_attention_string_for_display(piece_attention_vector, chess.square_name(from_square))
            thinking_lines = [
                f"Ply {ply}: Thinking about {san_move}...",
                f"Agent Value: {pre_move_value:.3f} | Stockfish: {stockfish_eval_str}",
                f"Agent Policy: {policy_str_display}",
                attention_str_display
            ]
            draw_frame(surface, board, piece_images, thinking_lines, piece_attention_vector, moving_piece_square=from_square)
            pygame.image.save(surface, os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png"))

            board.push(move)
            frame_count += 1
            
            is_check_after = board.is_check()
            is_mate_after = board.is_checkmate()
            _, post_move_value, _, _, _, _ = get_model_outputs_for_board(board, network, device)

            action_lines = [f"Ply {ply}: Played {san_move}", f"Resulting Agent Value: {post_move_value:.3f}", f"Value Change: {post_move_value - pre_move_value:+.3f}"]
            draw_frame(surface, board, piece_images, action_lines)
            pygame.image.save(surface, os.path.join(frame_dir_path, f"frame_{frame_count:04d}.png"))
            
            log_entry = {
                "ply": ply, "move_san": san_move, "board_fen_before": pre_move_fen,
                "move_uci": move.uci(),
                "move_characteristics": {"is_capture": is_capture, "is_check_after": is_check_after, "is_mate_after": is_mate_after},
                "agent_eval": {
                    "value_before": round(pre_move_value, 5), "value_after": round(post_move_value, 5),
                    "policy_before": [{"move": m, "prob": round(p, 5)} for m, p in pre_move_policy]
                },
                "stockfish_eval": stockfish_eval_str,
                "symmetric_attention": {
                    "piece_to_square": structured_ps_attention,
                    "square_to_piece": structured_sp_attention
                }
            }
            json_log_entries.append(log_entry)
            print(f"Processed ply {ply}: {san_move}")

    finally:
        pygame.quit()
        if stockfish_engine:
            stockfish_engine.quit()
            print("Stockfish engine terminated.")

    print("\nProcessing complete. Generating JSON Lines log file...")
    log_path = output_dir / f"{frame_dir_name}_analysis.jsonl"
    with open(log_path, 'w') as f:
        metadata = {
            "type": "analysis_metadata", "pgn_file": os.path.basename(args.pgn_path),
            "model_checkpoint": os.path.basename(str(model_path)), "stockfish_version": stockfish_path_from_config or "N/A",
            "analysis_timestamp_utc": datetime.utcnow().isoformat()
        }
        f.write(json.dumps(metadata) + '\n')
        
        for entry in json_log_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"-> Saved Structured Analysis Log to: {log_path}")

    gif_path = output_dir / f"{pgn_filename}.gif"
    loop_option = "-loop 0" if not args.no_loop else ""
    delay = 150
    print("\n✅ PNG frames and JSONL log file have been generated successfully.")
    print("To create the final GIF, please run the following command in your terminal:")
    print(f"\n--- \nconvert -delay {delay} {loop_option} '{frame_dir_path}/frame_*.png' '{gif_path}'\n--- \n")


def run_puzzle_generation(args):
    """
    Executes the puzzle generation mode, scanning PGNs for blunders.
    A blunder is defined as a move that causes a significant drop in evaluation
    compared to the best possible move in a position.
    """
    stockfish_path = config_params.get("STOCKFISH_PATH")
    if not stockfish_path or not os.path.exists(stockfish_path):
        print(f"Error: Stockfish path not set or found in config.py. Path was: {stockfish_path}")
        print("Puzzle generation requires a valid Stockfish engine.")
        return

    pgn_path = Path(args.pgn_path)
    if not pgn_path.exists():
        print(f"Error: PGN file not found at {pgn_path}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    BLUNDER_THRESHOLD_CP = args.blunder_threshold
    ANALYSIS_DEPTH = args.depth
    # A large centipawn value to represent a checkmate advantage
    MATE_SCORE = 100000 
    
    stockfish_engine = None
    puzzles_found_total = 0
    
    print(f"Starting blunder analysis on: {pgn_path}")
    print(f"Using Analysis Depth: {ANALYSIS_DEPTH} | Blunder Threshold: {BLUNDER_THRESHOLD_CP}cp")
    print(f"Output will be appended to: {output_path}")

    try:
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        with open(pgn_path, encoding='utf-8', errors='ignore') as pgn_file:
            game_num = 0
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                except Exception as e:
                    print(f"\nWarning: Failed to parse a game. Error: {e}. Skipping.")
                    continue
                    
                if game is None:
                    break
                    
                game_num += 1
                puzzles_in_game = 0
                print(f"  Scanning Game #{game_num} ({game.headers.get('White', '?')} vs {game.headers.get('Black', '?')})...", end='', flush=True)

                # Iterate through nodes to have access to board state before the move
                for node in game.mainline():
                    # Skip the root node which has no preceding move
                    if node.parent is None:
                        continue

                    board_before_move = node.parent.board()
                    move_played = node.move
                    
                    try:
                        # 1. Analyze the position BEFORE the move to find the best move and its eval
                        info_before = stockfish_engine.analyse(board_before_move, chess.engine.Limit(depth=ANALYSIS_DEPTH))
                        
                        if 'score' not in info_before or 'pv' not in info_before or not info_before['pv']:
                            continue
                        
                        best_move_found = info_before['pv'][0]
                        
                        # Don't create a puzzle if the move played was Stockfish's best move
                        if move_played == best_move_found:
                            continue

                        # Get the evaluation from the current player's perspective for the best possible move
                        eval_best_pov = info_before['score'].pov(board_before_move.turn)
                        
                        # 2. Get the evaluation AFTER the move was played
                        board_after_move = node.board()
                        info_after = stockfish_engine.analyse(board_after_move, chess.engine.Limit(depth=ANALYSIS_DEPTH - 2))

                        if 'score' not in info_after:
                            continue

                        # Evaluation is now from the other player's perspective.
                        # We get their POV and then take the .opponent() to flip it back.
                        eval_played_pov = info_after['score'].pov(board_after_move.turn).opponent()
                        
                        # 3. Compare evaluations in centipawns to find the drop
                        eval_best_cp = eval_best_pov.score(mate_score=MATE_SCORE)
                        eval_played_cp = eval_played_pov.score(mate_score=MATE_SCORE)
                        
                        eval_drop = eval_best_cp - eval_played_cp
                        
                        # 4. If the drop is a blunder, save the puzzle
                        if eval_drop >= BLUNDER_THRESHOLD_CP:
                            puzzles_in_game += 1
                            puzzles_found_total += 1
                            puzzle = {"fen": board_before_move.fen(), "best_move": best_move_found.uci()}
                            with open(output_path, 'a') as f:
                                f.write(json.dumps(puzzle) + '\n')
                                
                    except (chess.engine.EngineTerminatedError, chess.engine.EngineError, ValueError) as e:
                        print(f"\nStockfish analysis failed for a position: {e}")
                        # If the engine crashes, better to restart it
                        stockfish_engine.quit()
                        stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                
                if puzzles_in_game > 0:
                    print(f" Found {puzzles_in_game} blunder(s).")
                else:
                    print(" Done.")

    finally:
        if stockfish_engine:
            stockfish_engine.quit()
    
    print("\n-------------------------------------------------")
    print(f"Blunder analysis complete.")
    print(f"Found a total of {puzzles_found_total} new puzzles.")
    print(f"All puzzles have been appended to {output_path}")
    print("-------------------------------------------------")


def main():
    """
    Main function to parse arguments and dispatch to the correct mode.
    """
    parser = argparse.ArgumentParser(description="Generate analysis artifacts or tactical puzzles for a chess game.")
    
    # --- Arguments for both modes ---
    parser.add_argument("--pgn_path", required=True, help="Path to the PGN file to process.")
    
    # --- Mode selector ---
    parser.add_argument("--mode", type=str, choices=['analysis', 'puzzle'], default='analysis', 
                        help="Choose operation mode: 'analysis' for visual logs, 'puzzle' for blunder-based puzzle generation.")
    
    # --- Arguments specific to 'analysis' mode ---
    parser.add_argument("--model_path", type=str, default=None, 
                        help="(Analysis Mode) Optional path to a model checkpoint. If not provided, the latest is used.")
    parser.add_argument("--output_dir", default="analysis_output", 
                        help="(Analysis Mode) Directory to save frame and log artifacts.")
    parser.add_argument("--no-loop", action="store_true", 
                        help="(Analysis Mode) Provide this flag for the output GIF to play only once.")

    # --- Arguments specific to 'puzzle' mode ---
    parser.add_argument("--output", default="generated_puzzles.jsonl",
                        help="(Puzzle Mode) Path to the output JSONL file for appending puzzles. Default: generated_puzzles.jsonl")
    parser.add_argument("--blunder-threshold", type=int, default=200,
                        help="(Puzzle Mode) The minimum evaluation drop in centipawns to be considered a blunder. Default: 200 (2 pawns).")
    parser.add_argument("--depth", type=int, default=20,
                        help="(Puzzle Mode) The depth for the primary Stockfish analysis. Default: 20.")

    args = parser.parse_args()
    paths = get_paths()

    if args.mode == 'puzzle':
        run_puzzle_generation(args)
    elif args.mode == 'analysis':
        run_visual_analysis(args, paths)
    else:
        # This case should not be reachable due to 'choices' in argparse
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)


if __name__ == '__main__':
    main()