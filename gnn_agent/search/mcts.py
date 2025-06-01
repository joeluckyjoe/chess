import torch
import chess
from typing import Dict, Tuple, List # MODIFIED: Added List for policy type hint

# MODIFIED: Corrected import path based on your tree structure
from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input 
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from .mcts_node import MCTSNode


class MCTS:
    """
    Manages the Monte Carlo Tree Search algorithm.
    """
    # MODIFIED: __init__ to accept and store GNNDataConverter logic (via functions)
    def __init__(self, network: ChessNetwork, device: torch.device, c_puct: float = 1.41):
        """
        Initializes the MCTS search manager.
        Note: GNNDataConverter functionality is now directly imported and used.
        """
        self.network = network
        self.network.eval()  # Set the network to evaluation mode.
        self.device = device
        self.c_puct = c_puct
        self.root = None
        # self.data_converter = data_converter # No longer needed as an instance variable
                                            # if convert_to_gnn_input is a static/module function

    @torch.no_grad()
    def _expand_and_evaluate(self, node: MCTSNode, board: chess.Board):
        """
        Expands a leaf node, evaluates it with the neural network, and backpropagates.
        """
        # MODIFIED: Direct use of imported convert_to_gnn_input
        gnn_input_tuple = convert_to_gnn_input(board, self.device)
        policy_logits, value_tensor = self.network(*gnn_input_tuple)

        policy_probs = torch.softmax(policy_logits, dim=0) # Corrected dim from previous step

        legal_moves = list(board.legal_moves)
        policy_priors = {}
        if not legal_moves: # Terminal node with no moves, or stalemate
            self._backpropagate(node, value_tensor.item()) # Backpropagate game outcome if terminal
            return

        for move in legal_moves:
            try:
                idx = move_to_index(move, board)
                policy_priors[move] = policy_probs[idx].item()
            except IndexError:
                # This can happen if num_actions in PolicyHead is smaller than max possible index
                print(f"Warning: Move {move.uci()} resulted in an out-of-bounds index from move_to_index. Setting prior to small value.")
                policy_priors[move] = 1e-9 # Small epsilon or handle differently

        child_turn = not board.turn # chess.WHITE is True, chess.BLACK is False
        node.expand(legal_moves, policy_priors, child_turn)
        self._backpropagate(node, value_tensor.item())

    def _backpropagate(self, node: MCTSNode, value: float):
        """
        Propagates the evaluation result back up the tree.
        """
        # MODIFIED: Adjust value based on whose turn it was at the node being updated
        current_node = node
        current_value = value
        while current_node:
            # The value is from the perspective of the player whose turn it is AT THE LEAF.
            # When backpropagating, if the parent's turn is different, the value is inverted.
            # This node's N and Q are updated based on the outcome from its child's perspective.
            # The value should be relative to the player whose turn it is for the *node being updated*.
            current_node.N += 1
            current_node.Q += current_value
            if current_node.parent: # Check if parent exists
                 # If the move leading to 'node' was made by the opponent of 'node.parent'
                 # then the value for 'node.parent' is -value.
                 # Assuming MCTSNode value is always from current player's perspective for that node's state.
                 # AlphaGo Zero stores W (total action value), not Q. Q = W/N.
                 # Value from NN is P(s_L wins). If s_L is opponent's turn, value is opponent's win prob.
                 # So, for parent, value is -v_opponent.
                 # If board.turn at the node matches the turn when the value was evaluated, it's direct.
                 # If value is from the perspective of the player to move at the *leaf*,
                 # then for the parent, the value should be inverted.
                current_value *= -1 # Invert value for the parent node (opponent's turn)
            current_node = current_node.parent


    # MODIFIED: run_search to return policy, best_move, and root_gnn_input_tuple
    def run_search(self, board: chess.Board, num_simulations: int) -> Tuple[Dict[chess.Move, float], chess.Move, Tuple]:
        """
        Executes the MCTS search for a given number of simulations.
        Returns the MCTS policy (Dict[move, prob]), the best move, 
        and the GNN input tuple (board_tensor) for the root state.
        """
        self.root = MCTSNode(parent=None, prior_p=1.0, board_turn_at_node=board.turn) # Store whose turn it is at root

        # Get GNN input for the root board state. This is the 'board_tensor' for training.
        # MODIFIED: Direct use of imported convert_to_gnn_input
        root_gnn_input_tuple = convert_to_gnn_input(board, self.device)

        # Initial expansion and evaluation of the root if it's a leaf (first time)
        if self.root.is_leaf() and not board.is_game_over():
             self._expand_and_evaluate(self.root, board.copy()) # Use original board for root expansion

        for _ in range(num_simulations):
            sim_board = board.copy()
            current_node = self.root

            # --- 1. Selection ---
            while not current_node.is_leaf():
                if not current_node.children: # Should not happen if not leaf, but safety check
                    break 
                best_move = max(current_node.children, key=lambda move: current_node.children[move].uct_value(self.c_puct))
                sim_board.push(best_move)
                current_node = current_node.children[best_move]

            # --- 2. & 3. Expansion & Evaluation ---
            if not sim_board.is_game_over():
                self._expand_and_evaluate(current_node, sim_board)
            else:
                # Determine value for terminal states from the perspective of the player whose turn it *would be*
                outcome = sim_board.outcome()
                value = 0.0
                if outcome:
                    if outcome.winner == chess.WHITE:
                        value = 1.0
                    elif outcome.winner == chess.BLACK:
                        value = -1.0
                # The value needs to be from the perspective of the player whose turn it is at current_node
                # If current_node.board_turn_at_node is Black and White won, value for current_node is -1.
                if current_node.board_turn_at_node != sim_board.turn and outcome and outcome.winner is not None: # If terminal, turn has flipped
                     value *= -1 # If it was Black's turn at current_node, and White (sim_board.turn) won, current_node gets -1
                
                self._backpropagate(current_node, value)


        # After all simulations, select the best move based on the highest visit count.
        if not self.root.children:
            return {}, None, root_gnn_input_tuple

        # Calculate MCTS policy from visit counts
        mcts_policy_dict: Dict[chess.Move, float] = {}
        total_visits_at_root_children = sum(child.N for child in self.root.children.values())

        # Create a policy for all legal moves, not just explored children
        # This requires knowing the full action space for the root.
        # For simplicity, base policy on children discovered.
        # A more complete policy would map all legal moves to probabilities.
        
        # Get all legal moves from the original root board state
        # This ensures the policy dictionary has keys for all possible actions from the root.
        root_legal_moves = list(board.legal_moves)
        action_space_size = get_action_space_size() # Get total number of possible actions
        
        # Initialize policy for all actions to 0
        # The policy for SelfPlay is Dict[chess.Move, float]
        # So we only need to care about legal moves from the current state 'board'

        if total_visits_at_root_children > 0:
            for move in root_legal_moves:
                if move in self.root.children:
                    mcts_policy_dict[move] = self.root.children[move].N / total_visits_at_root_children
                else:
                    mcts_policy_dict[move] = 0.0 # Move was legal but not explored/chosen by MCTS enough to be a child
        else: # No visits to children (e.g., num_simulations was too low or immediate terminal state)
             # Fallback to network's prior probabilities for the root (if root was expanded)
            if self.root.priors:
                prior_sum = sum(self.root.priors.values())
                if prior_sum > 0:
                    for move in root_legal_moves:
                        mcts_policy_dict[move] = self.root.priors.get(move, 0.0) / prior_sum
                else: # If priors are also zero/empty, assign uniform
                    if root_legal_moves:
                        prob = 1.0 / len(root_legal_moves)
                        for move in root_legal_moves:
                            mcts_policy_dict[move] = prob
            elif root_legal_moves: # No priors and no visits, assign uniform
                prob = 1.0 / len(root_legal_moves)
                for move in root_legal_moves:
                    mcts_policy_dict[move] = prob
            # else, mcts_policy_dict remains empty if no legal moves

        # Select best move (move with highest visit count from root)
        best_move = max(self.root.children, key=lambda move: self.root.children[move].N, default=None)

        return mcts_policy_dict, best_move, root_gnn_input_tuple