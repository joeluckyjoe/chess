import torch
import chess
import chess.pgn
from typing import Tuple, Dict, List, Optional

from stockfish import Stockfish
from ..neural_network.hybrid_transformer_model import HybridTransformerModel # MODIFIED
from ..search.mcts import MCTS
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input
from ..gamestate_converters.action_space_converter import move_to_index
from torch_geometric.data import Batch

def _get_mentor_move_and_score(mentor_engine: Stockfish, board: chess.Board) -> Tuple[Optional[chess.Move], float]:
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
    agent: HybridTransformerModel, # MODIFIED
    mentor_engine: Stockfish,
    board: chess.Board,
    search_manager: MCTS,
    num_simulations: int,
    in_guided_mode: bool,
    value_threshold: float,
    agent_color: bool,
) -> Tuple[Optional[chess.Move], bool, Dict]: # MODIFIED: Signature
    model_device = next(agent.parameters()).device
    agent.eval()

    gnn_data, cnn_data, _ = convert_to_gnn_input(board, model_device)
    gnn_batch = Batch.from_data_list([gnn_data])
    cnn_batch = cnn_data.unsqueeze(0)
    
    # REMOVED: hidden_state_batch logic
    
    # MODIFIED: Call network without hidden state
    policy_logits, value_tensor, _ = agent(gnn_batch, cnn_batch)
    value = value_tensor.item()
    
    mentor_move, mentor_score_cp = _get_mentor_move_and_score(mentor_engine, board)
    if mentor_move is None:
        return None, in_guided_mode, {}
    
    mentor_value = torch.tanh(torch.tensor(mentor_score_cp / 10.0)).item()
    agent_pov_mentor_value = mentor_value if board.turn == chess.WHITE else -mentor_value
    
    # MODIFIED: Call MCTS search without hidden state
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
    agent: HybridTransformerModel, # MODIFIED
    mentor_engine: Stockfish,
    search_manager: MCTS,
    agent_color_str: str,
    num_simulations: int = 100,
    value_threshold: float = 0.1,
    contempt_factor: float = 0.0
) -> Tuple[List[Tuple[str, Dict, float]], str]:
    board = chess.Board()
    training_examples = []
    
    agent_color = chess.WHITE if agent_color_str.lower() == 'white' else chess.BLACK
    in_guided_mode = True
    
    # REMOVED: All hidden_state initialization and management logic

    while not board.is_game_over(claim_draw=True):
        if board.turn == agent_color:
            # MODIFIED: Call _decide_agent_action without hidden state
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
        outcome_perspective = mentor_score_cp if board.turn == chess.WHITE else -mentor_score_cp
        outcome = torch.tanh(torch.tensor(outcome_perspective / 10.0)).item()
        
        if abs(outcome) < 1e-6:
            outcome = contempt_factor

        if board.turn == agent_color:
            training_examples.append((board.fen(), policy, outcome))

        board.push(move)

    pgn = chess.pgn.Game.from_board(board)
    pgn.headers["Event"] = "Guided Mentor Session"
    pgn.headers["Site"] = "Herstal, Wallonia, Belgium"
    pgn.headers["Date"] = "2025.07.18" # MODIFIED: Date
    pgn.headers["White"] = "Mentor (Stockfish)" if agent_color == chess.BLACK else "Agent (v109)"
    pgn.headers["Black"] = "Agent (v109)" if agent_color == chess.BLACK else "Mentor (Stockfish)"
    pgn.headers["Result"] = board.result(claim_draw=True)
    pgn_string = str(pgn)
    
    return training_examples, pgn_string