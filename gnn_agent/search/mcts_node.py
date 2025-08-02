import chess
import math
from typing import Dict, Optional, Any, List

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.
    Stores statistics for a specific board state.
    """
    def __init__(self, parent: Optional['MCTSNode'] = None, prior_p: float = 0.0, board_turn_at_node: Optional[chess.Color] = None):
        self._parent = parent
        self._children: Dict[chess.Move, MCTSNode] = {}
        self.N: int = 0      # Visit count
        self.Q: float = 0.0  # Total action value (from the perspective of the player to move at this node)
        self.P: float = prior_p  # Prior probability of selecting this action
        self.board_turn_at_node: Optional[chess.Color] = board_turn_at_node

    @property
    def parent(self) -> Optional['MCTSNode']:
        return self._parent

    @property
    def children(self) -> Dict[chess.Move, 'MCTSNode']:
        return self._children

    def is_leaf(self) -> bool:
        return not self._children

    def expand(self, legal_moves: List[chess.Move], policy_priors: Dict[chess.Move, float], child_nodes_turn: chess.Color):
        """
        Expands the node by creating children for all legal moves.
        """
        for move in legal_moves:
            if move in policy_priors:
                self._children[move] = MCTSNode(parent=self, prior_p=policy_priors[move], board_turn_at_node=child_nodes_turn)

    def q_value(self) -> float:
        """
        Calculates the mean action value Q(s,a) for this node.
        This value is from the perspective of the player whose turn it is AT THIS NODE.
        """
        if self.N == 0:
            return 0.0
        return self.Q / self.N

    def uct_value(self, cpuct: float) -> float:
        """
        Calculates the Upper Confidence Bound for Trees (UCT) value for this action.
        This is called on a child node from the perspective of its parent.
        """
        if self.parent is None:
            raise ValueError("UCT value is only calculated for child nodes.")

        # The Q-value of a child node is from the perspective of the player to move at that child.
        # To evaluate this move from the parent's perspective, we must negate the child's Q-value.
        # Example: If parent is White, child is Black. A high Q-value for Black (+0.8) is a low
        # value for White (-0.8).
        q_for_parent = -self.q_value()

        # The exploration term encourages visiting less-explored actions.
        exploration_term = cpuct * self.P * (math.sqrt(self.parent.N) / (1 + self.N))
        
        return q_for_parent + exploration_term