# FILENAME: gnn_agent/search/mcts.py
import torch
import chess
import numpy as np
from typing import Dict, Tuple, Deque, Union
import collections

from torch_geometric.data import Batch
from ..gamestate_converters.action_space_converter import move_to_index
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input
from ..neural_network.policy_value_model import PolicyValueModel
from .mcts_node import MCTSNode


class MCTS:
    def __init__(self, network: PolicyValueModel, device: torch.device,
                 batch_size: int, c_puct: float = 1.41,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        self.network = network
        self.network.eval()
        self.device = device
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root: MCTSNode = None
        self._pending_evaluations: Deque[Tuple[MCTSNode, chess.Board]] = collections.deque()

    def _backpropagate(self, node: MCTSNode, value: float):
        current_node = node
        current_value = value
        while current_node:
            current_node.N += 1
            current_node.Q += current_value
            if current_node.parent:
                current_value *= -1
            current_node = current_node.parent

    @torch.no_grad()
    def _expand_and_evaluate_batch(self):
        if not self._pending_evaluations:
            return

        nodes_to_process, boards_to_process = zip(*self._pending_evaluations)
        self._pending_evaluations.clear()

        gnn_data_list, cnn_data_list, _ = zip(*[convert_to_gnn_input(b, torch.device('cpu')) for b in boards_to_process])
        batched_gnn_data = Batch.from_data_list(list(gnn_data_list)).to(self.device)
        batched_cnn_data = torch.stack(cnn_data_list, 0).to(self.device)

        policy_logits_batch, value_batch = self.network(batched_gnn_data, batched_cnn_data)
        policy_probs_batch = torch.softmax(policy_logits_batch, dim=1)

        for i, node in enumerate(nodes_to_process):
            board = boards_to_process[i]
            policy_probs = policy_probs_batch[i]
            value = value_batch[i].item()

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                self._backpropagate(node, value)
                continue

            policy_priors = {move: policy_probs[move_to_index(move, board)].item() for move in legal_moves}
            prior_sum = sum(policy_priors.values())
            if prior_sum > 1e-6:
                for move in policy_priors:
                    policy_priors[move] /= prior_sum
            
            node.expand(legal_moves, policy_priors, not board.turn)
            self._backpropagate(node, value)

    def _add_dirichlet_noise(self, node: MCTSNode):
        if not node.children: return
        children_moves = sorted(node.children.keys(), key=lambda m: m.uci())
        children_nodes = [node.children[move] for move in children_moves]
        priors = np.array([child.P for child in children_nodes], dtype=np.float32)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
        noisy_priors = (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
        for i, child in enumerate(children_nodes):
            child.P = noisy_priors[i]

    @torch.no_grad()
    # --- MODIFIED: The function now returns the policy and the MCTS-derived value ---
    def run_search(self, board: chess.Board, num_simulations: int) -> Tuple[Dict[chess.Move, float], float]:
        self.root = MCTSNode(parent=None, prior_p=1.0, board_turn_at_node=board.turn)

        if not board.is_game_over():
            gnn_data, cnn_data, _ = convert_to_gnn_input(board, self.device)
            gnn_batch = Batch.from_data_list([gnn_data])
            cnn_batch = cnn_data.unsqueeze(0)
            
            policy_logits, value = self.network(gnn_batch, cnn_batch)
            
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0)
            value = value.item()

            legal_moves = list(board.legal_moves)
            policy_priors = {move: policy_probs[move_to_index(move, board)].item() for move in legal_moves}
            prior_sum = sum(policy_priors.values())
            if prior_sum > 1e-6:
                for move in policy_priors: policy_priors[move] /= prior_sum
            
            self.root.expand(legal_moves, policy_priors, not board.turn)
            self._backpropagate(self.root, value)
            self._add_dirichlet_noise(self.root)

        sims_done = 1
        while sims_done < num_simulations:
            num_to_run_now = min(self.batch_size, num_simulations - sims_done)
            for _ in range(num_to_run_now):
                sim_board = board.copy()
                current_node = self.root
                while not current_node.is_leaf():
                    if not current_node.children: break
                    best_move = max(current_node.children, key=lambda move: current_node.children[move].uct_value(self.c_puct))
                    sim_board.push(best_move)
                    current_node = current_node.children[best_move]
                
                if not sim_board.is_game_over():
                    self._pending_evaluations.append((current_node, sim_board))
                else:
                    outcome = sim_board.outcome()
                    term_value = 0.0
                    if outcome:
                        winner = outcome.winner
                        if winner is not None:
                            player_at_leaf = current_node.board_turn_at_node
                            term_value = 1.0 if winner == player_at_leaf else -1.0
                    self._backpropagate(current_node, term_value)
            
            self._expand_and_evaluate_batch()
            sims_done += num_to_run_now

        if not self.root.children: 
            return {}, 0.0

        total_visits = sum(child.N for child in self.root.children.values())
        policy = {move: child.N / total_visits for move, child in self.root.children.items()} if total_visits > 0 else {}
        
        # --- NEW: Calculate the MCTS-derived value of the root state ---
        mcts_value = self.root.Q / self.root.N if self.root.N > 0 else 0.0

        return policy, mcts_value

    def select_move(self, policy: Dict[chess.Move, float], temperature: float) -> chess.Move:
        if not policy: return None
        moves = list(policy.keys())
        visit_counts = np.array([policy[m] for m in moves])
        if temperature < 1e-4:
            return moves[np.argmax(visit_counts)]
        else:
            powered_visits = np.power(visit_counts, 1.0 / temperature)
            probabilities = powered_visits / np.sum(powered_visits)
            return np.random.choice(moves, p=probabilities)