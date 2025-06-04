import torch
import chess
import numpy as np
from typing import Dict, Tuple, List

from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from .mcts_node import MCTSNode


class MCTS:
    def __init__(self, network: ChessNetwork, device: torch.device, c_puct: float = 1.41, dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        self.network = network
        self.network.eval()
        self.device = device
        self.c_puct = c_puct
        self.root = None
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    @torch.no_grad()
    def _expand_and_evaluate(self, node: MCTSNode, board: chess.Board):
        gnn_input_tuple = convert_to_gnn_input(board, self.device)
        policy_logits, value_tensor = self.network(*gnn_input_tuple)
        policy_probs = torch.softmax(policy_logits, dim=0)
        legal_moves = list(board.legal_moves)
        policy_priors = {}
        if not legal_moves:
            self._backpropagate(node, value_tensor.item())
            return
        for move in legal_moves:
            try:
                idx = move_to_index(move, board)
                policy_priors[move] = policy_probs[idx].item()
            except IndexError:
                print(f"Warning: Move {move.uci()} resulted in an out-of-bounds index from move_to_index. Setting prior to small value.")
                policy_priors[move] = 1e-9
        prior_sum = sum(policy_priors.values())
        if prior_sum > 1e-6:
            for move in policy_priors:
                policy_priors[move] /= prior_sum
        else:
            num_legal_moves = len(legal_moves)
            if num_legal_moves > 0:
                uniform_prob = 1.0 / num_legal_moves
                for move in legal_moves:
                    policy_priors[move] = uniform_prob
        child_turn = not board.turn
        node.expand(legal_moves, policy_priors, child_turn)
        self._backpropagate(node, value_tensor.item())

    def _add_dirichlet_noise(self, node: MCTSNode):
        if not node.children:
            return
        children_moves = sorted(node.children.keys(), key=lambda m: m.uci())
        children_nodes = [node.children[move] for move in children_moves]
        # THIS IS THE FIX: Using .P instead of .prior_p
        priors = np.array([child.P for child in children_nodes])
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
        noisy_priors = (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
        for i, child in enumerate(children_nodes):
            # THIS IS THE FIX: Using .P instead of .prior_p
            child.P = noisy_priors[i]

    def _backpropagate(self, node: MCTSNode, value: float):
        current_node = node
        current_value = value
        while current_node:
            current_node.N += 1
            current_node.Q += current_value
            if current_node.parent:
                current_value *= -1
            current_node = current_node.parent

    def run_search(self, board: chess.Board, num_simulations: int) -> Tuple[Dict[chess.Move, float], chess.Move, Tuple]:
        self.root = MCTSNode(parent=None, prior_p=1.0, board_turn_at_node=board.turn)
        root_gnn_input_tuple = convert_to_gnn_input(board, self.device)
        if self.root.is_leaf() and not board.is_game_over():
            self._expand_and_evaluate(self.root, board.copy())
            self._add_dirichlet_noise(self.root)
        for _ in range(num_simulations):
            sim_board = board.copy()
            current_node = self.root
            while not current_node.is_leaf():
                if not current_node.children:
                    break
                best_move = max(current_node.children, key=lambda move: current_node.children[move].uct_value(self.c_puct))
                sim_board.push(best_move)
                current_node = current_node.children[best_move]
            if not sim_board.is__game_over():
                self._expand_and_evaluate(current_node, sim_board)
            else:
                outcome = sim_board.outcome()
                value = 0.0
                if outcome:
                    if outcome.winner == chess.WHITE:
                        value = 1.0
                    elif outcome.winner == chess.BLACK:
                        value = -1.0
                if current_node.board_turn_at_node != sim_board.turn and outcome and outcome.winner is not None:
                    value *= -1
                self._backpropagate(current_node, value)
        if not self.root.children:
            return {}, None, root_gnn_input_tuple
        mcts_policy_dict: Dict[chess.Move, float] = {}
        total_visits_at_root_children = sum(child.N for child in self.root.children.values())
        root_legal_moves = list(board.legal_moves)
        if total_visits_at_root_children > 0:
            for move in root_legal_moves:
                if move in self.root.children:
                    mcts_policy_dict[move] = self.root.children[move].N / total_visits_at_root_children
                else:
                    mcts_policy_dict[move] = 0.0
        else:
            # Fallback logic simplified for brevity
            if root_legal_moves:
                prob = 1.0 / len(root_legal_moves)
                for move in root_legal_moves:
                    mcts_policy_dict[move] = prob
        best_move = max(self.root.children, key=lambda move: self.root.children[move].N, default=None)
        return mcts_policy_dict, best_move, root_gnn_input_tuple