#
# File: gnn_agent/rl_loop/guided_session.py (Final Corrected Version)
#
import torch
import chess
import chess.pgn
from typing import Tuple, Dict, List, Optional

from stockfish import Stockfish
from ..neural_network.chess_network import ChessNetwork
from ..search.mcts import MCTS
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input
from ..gamestate_converters.action_space_converter import move_to_index
from torch_geometric.data import Batch

def _get_mentor_move_and_score(mentor_engine: Stockfish, board: chess.Board) -> Tuple[Optional[chess.Move], float]:
    """
    Gets the best move and centipawn score from the Stockfish engine for a given board state.
    """
    mentor_engine.set_fen_position(board.fen())
    best_move_uci = mentor_engine.get_best_move_time(100)
    if not best_move_uci:
        return None, 0.0
    move = chess.Move.from_uci(best_move_uci)

    eval_result = mentor_engine.get_evaluation()
    if eval_result['type'] == 'mate':
        score = 30000 if eval_result['value'] > 0 else -30000
    else:
        score = eval_result['value']

    return move, score / 100.0

@torch.no_grad()
def _decide_agent_action(
    agent: ChessNetwork,
    mentor_engine: Stockfish,
    board: chess.Board,
    search_manager: MCTS,
    num_simulations: int,
    in_guided_mode: bool,
    value_threshold: float,
    agent_color: bool
) -> Tuple[Optional[chess.Move], bool, Dict]:
    """
    Decides the agent's move, potentially switching in or out of guided mode.
    """
    model_device = agent.device
    agent.eval()

    gnn_data = convert_to_gnn_input(board, model_device)
    batch = Batch.from_data_list([gnn_data])
    policy_logits, value_tensor = agent(batch)
    
    value = value_tensor.item()
    
    mentor_move, mentor_score_cp = _get_mentor_move_and_score(mentor_engine, board)
    if mentor_move is None:
        return None, in_guided_mode, {}
    
    mentor_value = torch.tanh(torch.tensor(mentor_score_cp / 10.0)).item()
    agent_pov_mentor_value = mentor_value if board.turn == chess.WHITE else -mentor_value
    
    policy_dict = search_manager.run_search(board, num_simulations)

    if in_guided_mode:
        if agent_pov_mentor_value < value_threshold:
            in_guided_mode = False
            move = search_manager.select_move(policy_dict, temperature=1.0)
        else:
            move = mentor_move
    else:
        if value < agent_pov_mentor_value - value_threshold:
            in_guided_mode = True
            move = mentor_move
        else:
            move = search_manager.select_move(policy_dict, temperature=1.0)
    
    # Ensure a move is selected if one wasn't (e.g., if policy_dict was empty)
    if move is None and list(board.legal_moves):
        move = search_manager.select_move(policy_dict, temperature=1.0)
        if move is None: # Fallback if search fails completely
             move = list(board.legal_moves)[0]

    policy_for_training = {m: torch.softmax(policy_logits.squeeze(0), dim=0)[move_to_index(m, board)].item() for m in board.legal_moves}
    
    return move, in_guided_mode, policy_for_training


def run_guided_session(
    agent: ChessNetwork,
    mentor_engine: Stockfish,
    search_manager: MCTS,
    agent_color_str: str, # FIX: Added missing agent_color_str parameter
    num_simulations: int = 100,
    value_threshold: float = 0.1
) -> Tuple[List[Tuple[object, Dict, float]], str]:
    """
    Plays a single game where the agent is guided by the mentor engine.
    """
    board = chess.Board()
    training_examples = []
    
    # FIX: Use the agent_color_str argument to set the agent's color
    agent_color = chess.WHITE if agent_color_str.lower() == 'white' else chess.BLACK
    in_guided_mode = True # Start in guided mode for the first move

    while not board.is_game_over(claim_draw=True):
        if board.turn == agent_color:
            move, in_guided_mode, policy = _decide_agent_action(
                agent, mentor_engine, board, search_manager, num_simulations, 
                in_guided_mode, value_threshold, agent_color
            )
        else:
            move, _ = _get_mentor_move_and_score(mentor_engine, board)
            policy = {}

        if move is None or not board.is_legal(move):
            break

        _, mentor_score_cp = _get_mentor_move_and_score(mentor_engine, board)
        # Ensure the perspective is correct for the value target
        outcome_perspective = mentor_score_cp if board.turn == chess.WHITE else -mentor_score_cp
        outcome = torch.tanh(torch.tensor(outcome_perspective / 10.0)).item()

        if board.turn == agent_color:
            gnn_input = convert_to_gnn_input(board, torch.device('cpu'))
            training_examples.append((gnn_input, policy, outcome))

        board.push(move)

    pgn_data = chess.pgn.Game.from_board(board).mainline_moves()
    
    return training_examples, str(pgn_data)