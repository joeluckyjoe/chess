# mcts_node.py

import math
from typing import Dict, Optional, Any

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.
    Stores statistics for a specific board state.
    """
    def __init__(self, parent: Optional['MCTSNode'] = None, prior_p: float = 1.0):
        """
        Initializes a new MCTS Node.

        Args:
            parent (Optional['MCTSNode']): The parent node. None for the root node.
            prior_p (float): The prior probability of selecting this node's action,
                             as determined by the policy head of the neural network.
        """
        self._parent = parent
        self._children: Dict[Any, 'MCTSNode'] = {} # Maps a move to a child node

        self._visit_count = 0  # N(s,a)
        self._total_action_value = 0.0  # Q(s,a)
        self._prior_p = prior_p  # P(s,a)

    @property
    def parent(self) -> Optional['MCTSNode']:
        return self._parent

    @property
    def children(self) -> Dict[Any, 'MCTSNode']:
        return self._children

    @property
    def visit_count(self) -> int:
        return self._visit_count

    @property
    def total_action_value(self) -> float:
        return self._total_action_value

    @property
    def prior_p(self) -> float:
        return self._prior_p

    def is_leaf_node(self) -> bool:
        """Checks if the node has any children."""
        return not self._children

    def expand(self, move_priors: Dict[Any, float]):
        """
        Expands the node by creating children for all legal moves from this state.

        Args:
            move_priors (Dict[Any, float]): A dictionary mapping legal moves
                                             to their prior probabilities from the policy head.
        """
        for move, prob in move_priors.items():
            if move not in self._children:
                self._children[move] = MCTSNode(parent=self, prior_p=prob)

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
        if self._visit_count == 0:
            return 0.0
        return self._total_action_value / self._visit_count

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