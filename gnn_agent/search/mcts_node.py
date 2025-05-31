# mcts_node.py

import chess
import math
from typing import Dict, Optional, Any

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.
    Stores statistics for a specific board state.
    """
    def __init__(self, parent: 'MCTSNode' = None, prior_p: float = 0.0):
        """
        Initializes a node in the MCTS tree.

        Args:
            parent (MCTSNode, optional): The parent of this node. Defaults to None.
            prior_p (float, optional): The prior probability of selecting this node,
                                    as predicted by the policy head. Defaults to 0.0.
        """
        self._parent = parent
        self._children: Dict[chess.Move, MCTSNode] = {}

        # Core MCTS statistics
        self.N: int = 0  # Visit count
        self.Q: float = 0.0  # Total action value
        self.P: float = prior_p  # Prior probability

    @property
    def parent(self) -> Optional['MCTSNode']:
        return self._parent

    @property
    def children(self) -> Dict[Any, 'MCTSNode']:
        return self._children

    @property
    def visit_count(self) -> int:
        return self.N

    @property
    def total_action_value(self) -> float:
        return self._total_action_value

    @property
    def prior_p(self) -> float:
        return self.P

    def is_leaf(self) -> bool:
        """Checks if the node has any children."""
        return not self._children

    def expand(self, legal_moves, policy_priors: Dict):
        """
        Expands the node by creating children for all legal moves.

        Args:
            legal_moves: A list or generator of legal moves from the current position.
            policy_priors: A dictionary mapping moves to their prior probabilities
                        as predicted by the neural network.
        """
        for move in legal_moves:
            if move in policy_priors:
                self._children[move] = MCTSNode(parent=self, prior_p=policy_priors[move])

    def backpropagate(self, value: float):
        """
        Updates the node's statistics and propagates the value up to the root.

        Args:
            value (float): The value of the terminal state from the perspective
                        of the player who just moved.
        """
        current = self
        while current is not None:
            current.N += 1
            current.Q += value
            # The value for the parent is from the other player's perspective,
            # so we must negate it for the next level up.
            value = -value
            current = current.parent

    def update(self, value: float):
        """
        Updates the node's statistics after a simulation (backpropagation).

        Args:
            value (float): The value of the terminal state from the simulation,
                           typically from the perspective of the current player.
        """
        self._visit_count += 1
        self._total_action_value += value

    def q_value(self) -> float:
        """
        Calculates the mean action value Q(s,a).
        This is the average outcome of simulations passing through this node.
        """
        if self.N == 0:
            return 0.0
        return self.Q / self.N

    def uct_value(self, cpuct: float = 1.41) -> float:
        """
        Calculates the Upper Confidence Bound for Trees (UCT) value.
        This balances exploration and exploitation.

        Args:
            cpuct (float): The exploration-exploitation trade-off constant.

        Returns:
            float: The UCT value for this node.
        """
        if self.parent is None:
             # The root node should not have a UCT value in the same way,
             # as selection starts from its children.
             # However, returning a standard value can prevent errors if called.
            return self.q_value()

        # U = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploration_term = (self.prior_p * math.sqrt(self.parent.visit_count)) / (1 + self.visit_count)
        return self.q_value() + cpuct * exploration_term