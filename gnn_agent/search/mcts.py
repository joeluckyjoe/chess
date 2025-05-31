#
# gnn_agent/search/mcts.py (Corrected and Verified)
#
import torch
import chess
from typing import Dict

from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from gnn_agent.gamestate_converters.action_space_converter import move_to_index
from gnn_agent.search.mcts_node import MCTSNode

class MCTS:
    """
    Manages the Monte Carlo Tree Search algorithm.
    """
    def __init__(self, network: ChessNetwork, device: torch.device, c_puct: float = 1.41):
        """
        Initializes the MCTS search manager.
        """
        self.network = network
        self.network.eval()  # Set the network to evaluation mode.
        self.device = device
        self.c_puct = c_puct
        self.root = None

    @torch.no_grad()
    def _expand_and_evaluate(self, node: MCTSNode, board: chess.Board):
        """
        Expands a leaf node, evaluates it with the neural network, and backpropagates.
        """
        gnn_input = convert_to_gnn_input(board, self.device)
        policy_logits, value = self.network(*gnn_input)

        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0)

        legal_moves = list(board.legal_moves)
        policy_priors = {}
        for move in legal_moves:
            idx = move_to_index(move, board)
            policy_priors[move] = policy_probs[idx].item()

        node.expand(legal_moves, policy_priors)
        self._backpropagate(node, value.item())

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Propagates the evaluation result back up the tree.
        """
        if node:
            node.backpropagate(value)

    def run_search(self, board: chess.Board, num_simulations: int) -> chess.Move:
        """
        Executes the MCTS search for a given number of simulations.
        """
        self.root = MCTSNode(parent=None, prior_p=1.0)

        for _ in range(num_simulations):
            sim_board = board.copy()
            current_node = self.root

            # --- 1. Selection ---
            while not current_node.is_leaf():
                # Corrected: self.c_puct without underscore
                best_move = max(current_node.children, key=lambda move: current_node.children[move].uct_value(self.c_puct))
                sim_board.push(best_move)
                current_node = current_node.children[best_move]

            # --- 2. & 3. Expansion & Evaluation ---
            if not sim_board.is_game_over():
                self._expand_and_evaluate(current_node, sim_board)
            else:
                if sim_board.is_checkmate():
                    value = -1.0
                else:
                    value = 0.0
                self._backpropagate(current_node, value)

        # After all simulations, select the best move based on the highest visit count.
        if not self.root.children:
            # This can happen in a lost position where there are no legal moves.
            # In a real game, this means we've already lost.
            return None

        # Corrected: .N instead of .visit_count
        best_move = max(self.root.children, key=lambda move: self.root.children[move].N)
        return best_move