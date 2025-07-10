#
# File: gnn_agent/rl_loop/guided_session.py (Final, Verified Version for Phase BA)
#
import torch
import chess
import chess.pgn
from typing import Tuple, Dict, List

# --- FIX: Corrected all imports based on the actual file tree ---
from ..neural_network.chess_network import ChessNetwork
from ..search.mcts import MCTS  # Use the existing MCTS class
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input
from ..gamestate_converters.stockfish_communicator import MentorEngine
from torch_geometric.data import Batch

@torch.no_grad()
def _decide_agent_action(
    agent: ChessNetwork,
    mentor_engine: MentorEngine,
    board: chess.Board,
    mcts_instance: MCTS,  # FIX: Type hint changed to MCTS
    num_simulations: int,
    in_guided_mode: bool,
    value_threshold: float,
    agent_color: bool
) -> Tuple[chess.Move, bool, Dict]:
    """
    Decides the agent's move, potentially switching in or out of guided mode.
    """
    model_device = agent.device
    agent.eval()

    gnn_data = convert_to_gnn_input(board, model_device)
    batch = Batch.from_data_list([gnn_data])
    policy_logits, value_tensor = agent(batch)
    
    value = value_tensor.item()
    policy_probs = torch.softmax(policy_logits.squeeze(0), dim=0)

    mentor_move, mentor_score = mentor_engine.get_best_move_with_score(board)
    agent_pov_mentor_score = mentor_score if board.turn == chess.WHITE else -mentor_score

    # Guided Mode Logic
    if in_guided_mode:
        if agent_pov_mentor_score < value_threshold:
            in_guided_mode = False
            # FIX: Call select_move on the mcts_instance object
            policy_dict = mcts_instance.run_search(board, num_simulations)
            move = mcts_instance.select_move(policy_dict, temperature=1.0)
        else:
            move = mentor_move
    else:  # Not in guided mode
        if value < agent_pov_mentor_score - value_threshold:
            in_guided_mode = True
            move = mentor_move
        else:
            # FIX: Call select_move on the mcts_instance object
            policy_dict = mcts_instance.run_search(board, num_simulations)
            move = mcts_instance.select_move(policy_dict, temperature=1.0)

    # Use the raw policy from the initial network pass for training data
    action_converter = mcts_instance.action_converter # Assuming action_converter is an attribute of MCTS
    policy_for_training = {m: policy_probs[action_converter.move_to_index(m, board)].item() for m in board.legal_moves}
    
    return move, in_guided_mode, policy_for_training


def run_guided_session(
    agent: ChessNetwork,
    mentor_engine: MentorEngine,
    mcts_instance: MCTS,  # FIX: Type hint changed to MCTS
    num_simulations: int = 100,
    value_threshold: float = 0.1
) -> Tuple[List[Tuple[object, Dict, float]], str]:
    """
    Plays a single game where the agent is guided by the mentor engine.
    """
    board = chess.Board()
    training_examples = []
    
    agent_color = chess.WHITE
    in_guided_mode = False

    while not board.is_game_over(claim_draw=True):
        if board.turn == agent_color:
            move, in_guided_mode, policy = _decide_agent_action(
                agent, mentor_engine, board, mcts_instance, num_simulations, 
                in_guided_mode, value_threshold, agent_color
            )
        else:
            move, _ = mentor_engine.get_best_move_with_score(board)
            policy = {}

        if move is None or not board.is_legal(move):
            break

        _, mentor_score = mentor_engine.get_best_move_with_score(board)
        outcome = torch.tanh(torch.tensor(mentor_score / 10.0)).item()

        if board.turn == agent_color:
            gnn_input = convert_to_gnn_input(board, torch.device('cpu'))
            training_examples.append((gnn_input, policy, outcome))

        board.push(move)

    pgn_data = chess.pgn.Game.from_board(board).mainline_moves()
    
    return training_examples, str(pgn_data)