import torch
import chess
import numpy as np
from typing import Dict, Tuple, List, Deque
import collections

from torch_geometric.data import Batch
from ..gamestate_converters.action_space_converter import move_to_index
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input
from ..neural_network.chess_network import ChessNetwork
from .mcts_node import MCTSNode


class MCTS:
    """
    A high-performance Monte Carlo Tree Search implementation using batched tree parallelization.
    This class is designed to maximize GPU throughput by evaluating leaf nodes in large batches.
    """
    def __init__(self, network: ChessNetwork, device: torch.device,
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

        # Data is created on CPU first for efficiency.
        data_list = [convert_to_gnn_input(b, torch.device('cpu')) for b in boards_to_process]

        # 1. Manually collate Square Graph tensors
        square_x_list = [d.square_features for d in data_list]
        square_edge_list = [d.square_edge_index for d in data_list]
        
        csum_sq = torch.cumsum(torch.tensor([s.size(0) for s in square_x_list]), 0)
        csum_sq = torch.cat([torch.tensor([0]), csum_sq[:-1]])
        
        # --- FIX: All input tensors must be moved to self.device ---
        square_features = torch.cat(square_x_list, dim=0).to(self.device)
        square_edge_index = torch.cat([e + c for e, c in zip(square_edge_list, csum_sq)], dim=1).to(self.device)
        square_batch = torch.tensor([i for i, s in enumerate(square_x_list) for _ in range(s.size(0))], dtype=torch.long).to(self.device)

        # 2. Manually collate Piece Graph tensors
        piece_x_list = [d.piece_features for d in data_list]
        piece_edge_list = [d.piece_edge_index for d in data_list]
        piece_map_list = [d.piece_to_square_map for d in data_list]

        csum_pc = torch.cumsum(torch.tensor([p.size(0) for p in piece_x_list]), 0)
        csum_pc = torch.cat([torch.tensor([0]), csum_pc[:-1]])

        # --- FIX: All input tensors must be moved to self.device ---
        piece_features = torch.cat(piece_x_list, dim=0).to(self.device)
        piece_edge_index = torch.cat([e + c for e, c in zip(piece_edge_list, csum_pc)], dim=1).to(self.device)
        piece_batch = torch.tensor([i for i, p in enumerate(piece_x_list) for _ in range(p.size(0))], dtype=torch.long).to(self.device)
        piece_to_square_map = torch.cat([pm + c for pm, c in zip(piece_map_list, csum_sq)], dim=0).to(self.device)

        # 3. Create the padding mask
        max_pieces = max(p.size(0) for p in piece_x_list) if piece_x_list else 0
        batch_size = len(boards_to_process)
        # --- FIX: Ensure padding mask is created directly on the correct device ---
        piece_padding_mask = torch.ones((batch_size, max_pieces), dtype=torch.bool, device=self.device)
        if piece_x_list:
            for i, p_features in enumerate(piece_x_list):
                num_pieces = p_features.size(0)
                if num_pieces > 0:
                    piece_padding_mask[i, :num_pieces] = 0

        # 4. Perform the forward pass with correctly batched & located data
        policy_logits_batch, value_batch = self.network(
            square_features=square_features,
            square_edge_index=square_edge_index,
            square_batch=square_batch,
            piece_features=piece_features,
            piece_edge_index=piece_edge_index,
            piece_batch=piece_batch,
            piece_to_square_map=piece_to_square_map,
            piece_padding_mask=piece_padding_mask
        )

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
            else:
                uniform_prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    policy_priors[move] = uniform_prob

            child_turn = not board.turn
            node.expand(legal_moves, policy_priors, child_turn)
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

    def run_search(self, board: chess.Board, num_simulations: int) -> Dict[chess.Move, float]:
        self.root = MCTSNode(parent=None, prior_p=1.0, board_turn_at_node=board.turn)
        
        sims_done = 0
        if not board.is_game_over():
            self._pending_evaluations.append((self.root, board.copy()))
            self._expand_and_evaluate_batch()
            self._add_dirichlet_noise(self.root)
            sims_done = 1

        while sims_done < num_simulations:
            num_to_run = min(self.batch_size, num_simulations - sims_done)
            for _ in range(num_to_run):
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
                    value = 0.0
                    if outcome:
                        if outcome.winner == chess.WHITE: value = 1.0
                        elif outcome.winner == chess.BLACK: value = -1.0
                    if current_node.board_turn_at_node != sim_board.turn and outcome and outcome.winner is not None:
                        value *= -1
                    self._backpropagate(current_node, value)
            
            self._expand_and_evaluate_batch()
            sims_done += num_to_run
            
        if not self.root.children: return {}
        total_visits = sum(child.N for child in self.root.children.values())
        return {move: child.N / total_visits for move, child in self.root.children.items()} if total_visits > 0 else {}

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