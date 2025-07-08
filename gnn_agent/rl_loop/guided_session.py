# gnn_agent/rl_loop/guided_session.py

import chess
import chess.pgn
import torch
import logging
from typing import Tuple, Optional, TYPE_CHECKING

# Local project imports
from gnn_agent.rl_loop.training_data_manager import TrainingDataBuffer

# This block is only for type analysis, not for runtime
if TYPE_CHECKING:
    from stockfish import Stockfish
    from gnn_agent.search.mcts import MCTSManager
    from gnn_agent.neural_network.chess_network import ChessNetwork

logger = logging.getLogger(__name__)

def _decide_agent_action(
    agent: 'ChessNetwork',
    mentor_engine: 'Stockfish',
    board: chess.Board,
    search_manager: 'MCTSManager',
    in_guided_mode: bool,
    value_threshold: float,
    agent_color: bool
) -> Tuple[chess.Move, bool, Optional[torch.Tensor]]:
    """
    Determines the agent's action for a single turn, applying guided session logic.
    Returns a tuple of (move_to_play, new_in_guided_mode, policy_for_buffer).
    """
    original_policy, _ = search_manager.run_search(board)
    agent_move_idx = torch.argmax(original_policy).item()
    agent_move = agent.action_to_move(agent_move_idx, board)
    policy_for_buffer = original_policy

    if not in_guided_mode:
        return agent_move, False, policy_for_buffer

    # --- Guided Mode Logic ---
    mentor_engine.set_fen_position(board.fen())
    mentor_move_uci = mentor_engine.get_best_move_time(100)
    mentor_move = chess.Move.from_uci(mentor_move_uci)

    if agent_move == mentor_move:
        logger.info("Agent and mentor agree. Ending guided mode.")
        return agent_move, False, policy_for_buffer

    # --- Intervention Triggered ---
    logger.info(f"Intervention Triggered. Agent: {agent_move}, Mentor: {mentor_move}")
    move_to_play = mentor_move

    board_after_mentor_move = board.copy()
    board_after_mentor_move.push(move_to_play)

    mentor_engine.set_fen_position(board_after_mentor_move.fen())
    eval_result = mentor_engine.get_evaluation()
    
    if eval_result['type'] == 'mate':
        mate_sign = 1 if eval_result['value'] > 0 else -1
        if board_after_mentor_move.turn == chess.BLACK:
            mate_sign *= -1
        mentor_eval_cp = mate_sign * 30000
    else:
        mentor_eval_cp = eval_result['value']
        
    if agent_color == chess.BLACK:
        mentor_eval_cp *= -1
        
    mentor_value = torch.tanh(torch.tensor(mentor_eval_cp / 1000.0))

    with torch.no_grad():
        agent.network.eval()
        gnn_data, _ = agent.gnn_data_converter.convert(board_after_mentor_move)
        _, agent_value = agent.network(gnn_data.to(agent.device))
        agent_value = agent_value.squeeze()

    value_discrepancy = torch.abs(agent_value - mentor_value)
    logger.info(f"Value Check. Agent Value: {agent_value.item():.4f}, Discrepancy: {value_discrepancy.item():.4f}")

    new_in_guided_mode = value_discrepancy >= value_threshold
    if not new_in_guided_mode:
        logger.info("Agent value aligned with mentor. Ending guided mode.")

    return move_to_play, new_in_guided_mode, policy_for_buffer


def run_guided_session(agent: 'ChessNetwork', mentor_engine: 'Stockfish', search_manager: 'MCTSManager', value_threshold: float, agent_color_str: str, max_guided_moves=15):
    """
    Runs a full guided mentor game and returns training examples and PGN data.
    """
    game = chess.pgn.Game()
    game.headers["Event"] = "Guided Mentor Session"
    node = game

    board = game.board()
    agent_color = chess.WHITE if agent_color_str.lower() == 'white' else chess.BLACK
    
    game.headers["White"] = f"Agent (v{getattr(agent, 'version', 'N/A')})" if agent_color == chess.WHITE else "Mentor Engine"
    game.headers["Black"] = "Mentor Engine" if agent_color == chess.WHITE else f"Agent (v{getattr(agent, 'version', 'N/A')})"

    in_guided_mode = True
    guided_moves_count = 0
    
    training_data_buffer = TrainingDataBuffer()

    while not board.is_game_over(claim_draw=True):
        move = None
        if board.turn == agent_color:
            if guided_moves_count >= max_guided_moves and in_guided_mode:
                logger.warning("Max guided moves reached. Exiting guided mode.")
                in_guided_mode = False

            move, new_in_guided_mode, policy = _decide_agent_action(
                agent, mentor_engine, board, search_manager, in_guided_mode, value_threshold, agent_color
            )
            
            if in_guided_mode and new_in_guided_mode:
                guided_moves_count += 1
            in_guided_mode = new_in_guided_mode
            
            if policy is not None:
                training_data_buffer.add_transient(board, policy)
        else: # Opponent's turn
            mentor_engine.set_fen_position(board.fen())
            opponent_move_uci = mentor_engine.get_best_move_time(50)
            move = chess.Move.from_uci(opponent_move_uci)

        if move:
            node = node.add_variation(move)
            board.push(move)

    outcome = board.result(claim_draw=True)
    game.headers["Result"] = outcome
    final_examples = training_data_buffer.finalize_game(outcome, agent_color)
    pgn_data = str(game)
    
    logger.info(f"Guided session finished. Outcome: {outcome}")
    return final_examples, pgn_data