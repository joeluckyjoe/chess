# test_mcts_node.py

import unittest
from mcts_node import MCTSNode # Assumes the class is in mcts_node.py

class TestMCTSNode(unittest.TestCase):

    def test_initialization(self):
        """Test that a new node is initialized with correct default values."""
        node = MCTSNode()
        self.assertIsNone(node.parent)
        self.assertEqual(node.prior_p, 1.0)
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.total_action_value, 0.0)
        self.assertEqual(node.q_value(), 0.0)
        self.assertTrue(node.is_leaf_node())
        self.assertEqual(len(node.children), 0)

    def test_initialization_with_parent_and_prior(self):
        """Test initialization with a specified parent and prior probability."""
        parent = MCTSNode()
        child = MCTSNode(parent=parent, prior_p=0.5)
        self.assertIs(child.parent, parent)
        self.assertEqual(child.prior_p, 0.5)

    def test_update(self):
        """Test the update method for visit count and action value."""
        node = MCTSNode()
        node.update(0.8)
        self.assertEqual(node.visit_count, 1)
        self.assertEqual(node.total_action_value, 0.8)
        self.assertEqual(node.q_value(), 0.8)

        node.update(-0.5)
        self.assertEqual(node.visit_count, 2)
        self.assertAlmostEqual(node.total_action_value, 0.3)
        self.assertAlmostEqual(node.q_value(), 0.15)

    def test_expand(self):
        """Test that the node correctly creates child nodes."""
        node = MCTSNode()
        # In a real scenario, moves would be chess.Move objects. Here we use strings.
        move_priors = {'e4': 0.7, 'd4': 0.3}
        node.expand(move_priors)

        self.assertFalse(node.is_leaf_node())
        self.assertEqual(len(node.children), 2)
        self.assertIn('e4', node.children)
        self.assertIn('d4', node.children)

        child_e4 = node.children['e4']
        self.assertIsInstance(child_e4, MCTSNode)
        self.assertIs(child_e4.parent, node)
        self.assertEqual(child_e4.prior_p, 0.7)

    def test_uct_value(self):
        """Test the UCT calculation."""
        parent = MCTSNode()
        parent.update(1) # Simulate parent being visited once
        parent.update(1) # and again
        self.assertEqual(parent.visit_count, 2)

        move_priors = {'e4': 0.6}
        parent.expand(move_priors)
        child_e4 = parent.children['e4']

        # First update for child
        child_e4.update(-1.0) # Game outcome is -1 from this node's perspective
        self.assertEqual(child_e4.visit_count, 1)
        self.assertEqual(child_e4.q_value(), -1.0)

        # UCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        # UCT = -1.0 + 1.41 * 0.6 * sqrt(2) / (1 + 1)
        # UCT = -1.0 + 1.41 * 0.6 * 1.4142 / 2
        # UCT = -1.0 + 0.598
        # UCT approx = -0.402
        expected_uct = -1.0 + 1.41 * 0.6 * (2**0.5) / 2
        self.assertAlmostEqual(child_e4.uct_value(cpuct=1.41), expected_uct)

if __name__ == '__main__':
    unittest.main()