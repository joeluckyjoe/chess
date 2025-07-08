# gnn_agent/rl_loop/guided_session.py

import chess
import chess.pgn
import torch
import logging
from typing import Tuple, Optional, TYPE_CHECKING, Dict

from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input

if TYPE_CHECKING:
    from stockfish import Stockfish
    from gnn_agent.search.mcts import MCTS
    from gnn_agent.neural_network.chess_network import ChessNetwork

logger = logging.getLogger(__name__)

def _decide_agent_action(
    agent: 'ChessNetwork',
    mentor_engine: 'Stockfish',
    board: chess.Board,
    search_manager: 'MCTS',
    num_simulations: int,
    in_guided_mode: bool,
    value_threshold: float,
    agent_color: bool
) -> Tuple[chess.Move, bool, Optional[Dict[chess.Move, float]]]:
    """
    Determines the agent's action for a single turn, applying guided session logic.
    """
    policy_dict = search_manager.run_search(board, num_simulations)
    agent_move = search_manager.select_move(policy_dict, temperature=0.0)
    
    if agent_move is None:
        return None, False, None

    if not in_guided_mode:
        return agent_move, False, policy_dict

    mentor_engine.set_fen_position(board.fen())
    mentor_move = chess.Move.from_uci(mentor_engine.get_best_move_time(100))

    if agent_move == mentor_move:
        logger.info("Agent and mentor agree. Ending guided mode.")
        return agent_move, False, policy_dict

    logger.info(f"Intervention Triggered. Agent: {agent_move}, Mentor: {mentor_move}")
    move_to_play = mentor_move
    
    board_after_mentor_move = board.copy()
    board_after_mentor_move.push(move_to_play)

    mentor_engine.set_fen_position(board_after_mentor_move.fen())
    eval_result = mentor_engine.get_evaluation()
    
    if eval_result['type'] == 'mate':
        mate_sign = 1 if eval_result['value'] > 0 else -1
        if board_after_mentor_move.turn == chess.BLACK: mate_sign *= -1
        mentor_eval_cp = mate_sign * 30000
    else:
        mentor_eval_cp = eval_result['value']
        
    if agent_color == chess.BLACK: mentor_eval_cp *= -1
    mentor_value = torch.tanh(torch.tensor(mentor_eval_cp / 1000.0))

    with torch.no_grad():
        agent.eval()
        model_device = next(agent.parameters()).device
        gnn_data = convert_to_gnn_input(board_after_mentor_move, model_device)
        
        kwargs = gnn_data.to_dict()
        kwargs['square_batch'] = torch.zeros(gnn_data.square_features.size(0), dtype=torch.long, device=model_device)
        kwargs['piece_batch'] = torch.zeros(gnn_data.piece_features.size(0), dtype=torch.long, device=model_device)
        kwargs['piece_padding_mask'] = torch.zeros((1, gnn_data.piece_features.size(0)), dtype=torch.bool, device=model_device)
        
        _, agent_value = agent(**kwargs)
        agent_value = agent_value.squeeze()

    value_discrepancy = torch.abs(agent_value - mentor_value)
    logger.info(f"Value Check. Agent Value: {agent_value.item():.4f}, Discrepancy: {value_discrepancy.item():.4f}")

    new_in_guided_mode = value_discrepancy >= value_threshold
    if not new_in_guided_mode:
        logger.info("Agent value aligned with mentor. Ending guided mode.")

    return move_to_play, new_in_guided_mode, policy_dict


def run_guided_session(agent: 'ChessNetwork', mentor_engine: 'Stockfish', search_manager: 'MCTS', num_simulations: int, value_threshold: float, agent_color_str: str, max_guided_moves=15):
    game = chess.pgn.Game()
    game.headers["Event"] = "Guided Mentor Session"
    node = game
    board = game.board()
    agent_color = chess.WHITE if agent_color_str.lower() == 'white' else chess.BLACK
    game.headers["White"] = f"Agent (v{getattr(agent, 'version', 'N/A')})" if agent_color == chess.WHITE else "Mentor Engine"
    game.headers["Black"] = "Mentor Engine" if agent_color == chess.WHITE else f"Agent (v{getattr(agent, 'version', 'N/A')})"
    in_guided_mode, guided_moves_count, transient_examples = True, 0, []

    while not board.is_game_over(claim_draw=True):
        move = None
        if board.turn == agent_color:
            if guided_moves_count >= max_guided_moves and in_guided_mode:
                logger.warning("Max guided moves reached. Exiting guided mode.")
                in_guided_mode = False

            move, new_in_guided_mode, policy = _decide_agent_action(agent, mentor_engine, board, search_manager, num_simulations, in_guided_mode, value_threshold, agent_color)
            if move is None:
                logger.error("MCTS search failed to return a move in guided session. Ending game.")
                break
            if in_guided_mode and new_in_guided_mode: guided_moves_count += 1
            in_guided_mode = new_in_guided_mode
            if policy is not None:
                # --- FINAL CORRECTION: Store the FEN string, not the board object ---
                transient_examples.append({'fen': board.fen(), 'policy': policy})
        else:
            mentor_engine.set_fen_position(board.fen())
            move = chess.Move.from_uci(mentor_engine.get_best_move_time(50))
        if move:
            node = node.add_variation(move)
            board.push(move)

    outcome_value = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}.get(board.result(claim_draw=True), 0)
    final_outcome_value = -outcome_value if agent_color == chess.BLACK else outcome_value
    
    # Unpack the FEN string from the dictionary
    final_examples = [
        (ex['fen'], ex['policy'], final_outcome_value)
        for ex in transient_examples
    ]

    game.headers["Result"] = board.result(claim_draw=True)
    logger.info(f"Guided session finished. Outcome: {game.headers['Result']}")
    return final_examples, str(game)