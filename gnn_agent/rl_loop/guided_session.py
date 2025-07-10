#
# File: gnn_agent/rl_loop/guided_session.py (Corrected for Phase BA)
#
import torch
import chess
import logging
from typing import Tuple, Dict, List

from ..neural_network.chess_network import ChessNetwork
from ..search.mcts import MCTS
from ..search.search_manager import SearchManager
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input
from ...engine.mentor_engine import MentorEngine
from ...utils.log_utils import get_file_handler

# --- FIX: Import Batch for proper single-item batching ---
from torch_geometric.data import Batch

# Configure logging for this module
# guided_session_logger = logging.getLogger("guided_session")
# if not guided_session_logger.handlers:
#     # Add a file handler if one doesn't exist
#     log_file_path = get_file_handler("supervisor_log.txt").baseFilename
#     file_handler = logging.FileHandler(log_file_path)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     guided_session_logger.addHandler(file_handler)
#     guided_session_logger.setLevel(logging.INFO)


def _get_agent_policy_and_value(agent: MCTS, board: chess.Board, num_simulations: int) -> Tuple[Dict[chess.Move, float], float]:
    """
    Runs MCTS search to get the agent's policy and returns the root node's value.
    """
    policy = agent.run_search(board, num_simulations)
    
    # The value of a state is the expected outcome from the current player's perspective.
    # MCTS node Q-value is calculated from the perspective of the node's parent.
    # The root node has no parent, but its children's Q-values are from the root's perspective.
    # We can average the Q-values of the children weighted by their visit counts (policy)
    # to estimate the state's value.
    value = 0.0
    if agent.root and agent.root.children:
        total_visits = sum(child.N for child in agent.root.children.values())
        if total_visits > 0:
            value = sum(
                child.Q * (child.N / total_visits) for child in agent.root.children.values()
            )

    return policy, value

@torch.no_grad()
def _decide_agent_action(agent: ChessNetwork, mentor_engine: MentorEngine, board: chess.Board, search_manager: SearchManager, num_simulations: int, in_guided_mode: bool, value_threshold: float, agent_color: bool) -> Tuple[chess.Move, bool, Dict]:
    """
    Decides the agent's move, potentially switching in or out of guided mode.
    This function contains the core logic for the Guided Mentor Session.
    """
    model_device = agent.device
    agent.eval()

    # --- FIX: The entire manual data preparation is removed ---
    # 1. Convert the current board state to a HeteroData object.
    gnn_data = convert_to_gnn_input(board, model_device)
    
    # 2. Use Batch.from_data_list to create a batch of size 1.
    #    This ensures the data has the correct batch attributes for the network.
    batch = Batch.from_data_list([gnn_data])

    # 3. Get the policy and value from the network using the single batch object.
    policy_logits, value_tensor = agent(batch)
    
    # --- End of FIX ---

    value = value_tensor.item()
    policy_probs = torch.softmax(policy_logits.squeeze(0), dim=0)

    # The rest of the function remains the same as it uses the correctly-derived outputs.
    mentor_move, mentor_score = mentor_engine.get_best_move_with_score(board)

    agent_pov_mentor_score = mentor_score if board.turn == chess.WHITE else -mentor_score

    # Guided Mode Logic
    if in_guided_mode:
        if agent_pov_mentor_score < value_threshold:
            # Agent has caught up, switch back to self-play for this move
            in_guided_mode = False
            # guided_session_logger.info(f"GUIDANCE OFF: Agent value ({value:.4f}) meets mentor value ({agent_pov_mentor_score:.4f}). Agent chooses own move.")
            move = search_manager.select_move(board, num_simulations)
        else:
            # Agent still needs guidance
            move = mentor_move
            # guided_session_logger.info(f"GUIDANCE ON: Agent value ({value:.4f}) below mentor ({agent_pov_mentor_score:.4f}). Playing mentor move: {move.uci()}")
    else: # Not in guided mode
        if value < agent_pov_mentor_score - value_threshold:
            # Agent has blundered, switch to guided mode
            in_guided_mode = True
            move = mentor_move
            # guided_session_logger.info(f"GUIDANCE ON: Agent value ({value:.4f}) fell below threshold ({agent_pov_mentor_score - value_threshold:.4f}). Playing mentor move: {move.uci()}")
        else:
            # Agent is doing fine, continue self-play
            move = search_manager.select_move(board, num_simulations)

    # Use the raw policy from the initial network pass for training data
    policy_for_training = {m: policy_probs[search_manager.action_converter.move_to_index(m, board)].item() for m in board.legal_moves}
    
    return move, in_guided_mode, policy_for_training


def run_guided_session(
    agent: ChessNetwork,
    mentor_engine: MentorEngine,
    search_manager: SearchManager,
    num_simulations: int = 100,
    value_threshold: float = 0.1
) -> Tuple[List[Tuple[object, Dict, float]], str]:
    """
    Plays a single game where the agent is guided by the mentor engine.
    The agent plays its own moves until its evaluated position value drops
    significantly below the mentor's evaluation, at which point the mentor's
    moves are played instead until the agent 'catches up'.
    """
    board = chess.Board()
    training_examples = []
    
    # The agent's color is randomized for each guided session
    agent_color = chess.WHITE # np.random.choice([chess.WHITE, chess.BLACK])
    in_guided_mode = False

    while not board.is_game_over(claim_draw=True):
        if board.turn == agent_color:
            move, in_guided_mode, policy = _decide_agent_action(agent, mentor_engine, board, search_manager, num_simulations, in_guided_mode, value_threshold, agent_color)
        else:
            # Mentor plays as the opponent
            move, _ = mentor_engine.get_best_move_with_score(board)
            policy = {} # No policy needed for opponent moves

        if move is None or not board.is_legal(move):
            # guided_session_logger.warning(f"Illegal or None move generated: {move}. Ending game.")
            break

        # For training, always use the mentor's evaluation as the ground truth outcome
        _, mentor_score = mentor_engine.get_best_move_with_score(board)
        outcome = torch.tanh(torch.tensor(mentor_score / 10.0)).item() # Scale and clamp value

        # Create training example from the agent's perspective
        if board.turn == agent_color:
            gnn_input = convert_to_gnn_input(board, torch.device('cpu'))
            # The agent learns from the policy it would have played,
            # but the outcome is always the objective mentor evaluation.
            training_examples.append((gnn_input, policy, outcome))

        board.push(move)

    pgn_data = chess.pgn.Game.from_board(board).mainline_moves()
    # guided_session_logger.info(f"Guided session finished. Result: {board.result(claim_draw=True)}. PGN: {pgn_data}")
    
    return training_examples, str(pgn_data)