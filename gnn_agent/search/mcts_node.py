# mcts_node.py

import chess
import math
from typing import Dict, Optional, Any, List, TYPE_CHECKING # Added List

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.
    Stores statistics for a specific board state.
    """
    # MODIFIED: Added board_turn_at_node and type hinted parent/children correctly
    def __init__(self, parent: Optional['MCTSNode'] = None, prior_p: float = 0.0, board_turn_at_node: Optional[chess.Color] = None):
        """
        Initializes a node in the MCTS tree.

        Args:
            parent (MCTSNode, optional): The parent of this node. Defaults to None.
            prior_p (float, optional): The prior probability of selecting this node,
                                     as predicted by the policy head. Defaults to 0.0.
            board_turn_at_node (chess.Color, optional): The player (chess.WHITE or chess.BLACK)
                                                      whose turn it is at this node's state.
                                                      Defaults to None.
        """
        self._parent = parent
        self._children: Dict[chess.Move, MCTSNode] = {} # Type hint MCTSNode for value

        # Core MCTS statistics
        self.N: int = 0      # Visit count
        self.Q: float = 0.0  # Total action value (sum of values from simulations passing through here)
        self.P: float = prior_p  # Prior probability of selecting this action (from policy network)
        
        # ADDED: Store whose turn it is for this node's state
        self.board_turn_at_node: Optional[chess.Color] = board_turn_at_node

    @property
    def parent(self) -> Optional['MCTSNode']:
        return self._parent

    @property
    def children(self) -> Dict[chess.Move, 'MCTSNode']: # Corrected type hint
        return self._children

    # REMOVED: Redundant visit_count property (use self.N directly)
    # @property
    # def visit_count(self) -> int:
    #     return self.N

    # REMOVED: Redundant total_action_value property (use self.Q directly)
    # @property
    # def total_action_value(self) -> float:
    #    return self.Q # Assuming Q is total action value, not mean

    # REMOVED: Redundant prior_p property (use self.P directly)
    # @property
    # def prior_p(self) -> float:
    #    return self.P

    def is_leaf(self) -> bool:
        """Checks if the node has any children (i.e., has not been expanded yet)."""
        return not self._children

    def expand(self, legal_moves: List[chess.Move], policy_priors: Dict[chess.Move, float], current_board_turn: chess.Color): # MODIFIED
        """
        Expands the node by creating children for all legal moves.

        Args:
            legal_moves: A list of legal chess.Move objects from the current position.
            policy_priors: A dictionary mapping legal_moves to their prior probabilities.
            current_board_turn: The player whose turn it is *after* one of these moves is made
                                (i.e., the turn of the child nodes).
        """
        for move in legal_moves:
            if move in policy_priors:
                # The board_turn_at_node for a child is the turn *after* the move is made
                self._children[move] = MCTSNode(parent=self, prior_p=policy_priors[move], board_turn_at_node=current_board_turn)
            # else:
                # Optionally handle moves that are legal but not in policy_priors (e.g., assign small prior)
                # print(f"Warning: Move {move.uci()} not in policy_priors during expansion.")


    def backpropagate(self, value: float):
        """
        Updates the node's statistics and propagates the value up to the root.
        The 'value' is from the perspective of the player whose turn it is at the node
        where the simulation ended (or was evaluated by the network).

        Args:
            value (float): The value of the terminal/evaluated state.
                           +1 if the current player (at the point of evaluation) wins,
                           -1 if they lose, 0 for a draw.
        """
        current_node = self
        # The 'value' passed in is from the perspective of the player whose turn it was at the
        # leaf node (or terminal state) that was evaluated.
        # As we go up, we need to ensure the value is interpreted correctly for current_node.parent.
        # If current_node.board_turn_at_node is WHITE, value is from WHITE's perspective.
        # If current_node.parent.board_turn_at_node is BLACK, then for parent, value must be -value.

        current_value_perspective = value # Value from the perspective of the child node that triggered backprop
        
        while current_node is not None:
            current_node.N += 1
            # If the player whose turn it is at 'current_node' is DIFFERENT from the player
            # whose perspective 'current_value_perspective' represents, we flip the value.
            # This happens automatically if current_value_perspective is inverted at each step up.
            current_node.Q += current_value_perspective
            
            # Invert the value for the parent, as it's now from the other player's perspective
            current_value_perspective = -current_value_perspective 
            current_node = current_node.parent

    # REMOVED: update method, as backpropagate handles N and Q directly.
    # def update(self, value: float):
    #     ...

    def q_value(self) -> float:
        """
        Calculates the mean action value Q(s,a) for this node (if it's a child).
        If this node is the root, this is V(s_root).
        This value is from the perspective of the player whose turn it was AT THIS NODE.
        """
        if self.N == 0:
            return 0.0
        return self.Q / self.N

    def uct_value(self, cpuct: float) -> float: # MODIFIED: cpuct should be passed
        """
        Calculates the Upper Confidence Bound for Trees (UCT) value for this node (action).
        This is called on children of a node to select which child to traverse.
        Value is from the perspective of the player whose turn it is at the PARENT node.

        Args:
            cpuct (float): The exploration-exploitation trade-off constant.

        Returns:
            float: The UCT value for this node.
        """
        if self.parent is None: # Should not happen if called on a child
            # This node is a root node or has no parent, UCT is not typically defined
            # in the same way. This function is meant to be called on children during selection.
            # Raising an error or returning a sensible default.
            raise ValueError("UCT value is typically calculated for child nodes.")

        # Q(s,a) is from the perspective of the current player at the parent node.
        # self.q_value() is from the perspective of player at *this* node (child).
        # Since the parent made a move to reach this child, this child represents the state
        # *after* parent's move, so it's opponent's turn.
        # Therefore, Q_for_parent = -self.q_value() (because self.q_value is from child's perspective)
        
        q_for_parent = -self.q_value() # Value of this child state from the parent's perspective

        exploration_term = cpuct * self.P * (math.sqrt(self.parent.N) / (1 + self.N))
        
        return q_for_parent + exploration_term