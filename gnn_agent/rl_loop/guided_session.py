#
# File: gnn_agent/rl_loop/guided_session.py (Corrected for GNN+CNN Hybrid)
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
    model_device = next(agent.parameters()).device
    agent.eval()

    # --- MODIFICATION FOR GNN+CNN HYBRID MODEL ---
    # 1. Unpack the GNN and CNN data from the converter.
    gnn_data, cnn_data = convert_to_gnn_input(board, torch.device('cpu'))

    # 2. Batch the single GNN graph and move to the device.
    batched_gnn_data = Batch.from_data_list([gnn_data]).to(model_device)

    # 3. Add a batch dimension to the CNN tensor and move to the device.
    batched_cnn_data = cnn_data.unsqueeze(0).to(model_device)

    # 4. Perform the forward pass with both inputs.
    policy_logits, value_tensor = agent(batched_gnn_data, batched_cnn_data)
    # --- END MODIFICATION ---
    
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
    
    if move is None and list(board.legal_moves):
        if not policy_dict:
            policy_dict = search_manager.run_search(board, num_simulations)
        move = search_manager.select_move(policy_dict, temperature=1.0)
        if move is None:
            move = list(board.legal_moves)[0]

    policy_for_training = {m: torch.softmax(policy_logits.squeeze(0), dim=0)[move_to_index(m, board)].item() for m in board.legal_moves}
    
    return move, in_guided_mode, policy_for_training


def run_guided_session(
    agent: ChessNetwork,
    mentor_engine: Stockfish,
    search_manager: MCTS,
    agent_color_str: str,
    num_simulations: int = 100,
    value_threshold: float = 0.1
) -> Tuple[List[Tuple[str, Dict, float]], str]:
    """
    Plays a single game where the agent is guided by the mentor engine.
    """
    board = chess.Board()
    training_examples = []
    
    agent_color = chess.WHITE if agent_color_str.lower() == 'white' else chess.BLACK
    in_guided_mode = True

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

        # Get the objective evaluation of the position *before* the move
        _, mentor_score_cp = _get_mentor_move_and_score(mentor_engine, board)
        outcome_perspective = mentor_score_cp if board.turn == chess.WHITE else -mentor_score_cp
        outcome = torch.tanh(torch.tensor(outcome_perspective / 10.0)).item()

        # Generate training example for the agent's turn
        if board.turn == agent_color:
            training_examples.append((board.fen(), policy, outcome))

        board.push(move)

    pgn = chess.pgn.Game.from_board(board)
    pgn_string = str(pgn.mainline_moves())
    
    return training_examples, pgn_string