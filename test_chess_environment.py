# test_chess_environment.py
import unittest
from chess_environment import ChessEnvironment # Assuming your file is named chess_environment.py
import chess

class TestChessEnvironment(unittest.TestCase):

    def test_initial_board_state_fen(self):
        """
        Tests that the board initializes to the standard starting FEN.
        """
        env = ChessEnvironment()
        # Standard FEN for starting position
        expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.assertEqual(env.get_current_state_fen(), expected_fen)

    def test_initial_legal_moves(self):
        """
        Tests that the initial legal moves are correct (20 possible moves).
        """
        env = ChessEnvironment()
        legal_moves = env.get_legal_moves()
        self.assertEqual(len(legal_moves), 20)
        # Spot check a few common starting moves
        self.assertIn("g1f3", legal_moves)
        self.assertIn("e2e4", legal_moves)
        self.assertIn("d2d4", legal_moves)
        self.assertNotIn("e1g1", legal_moves) # Castling not legal yet from start

    def test_initial_player(self):
        """
        Tests that the initial player is White.
        """
        env = ChessEnvironment()
        self.assertTrue(env.get_current_player(), "Initial player should be White")

    def test_apply_valid_move(self):
        env = ChessEnvironment()
        initial_fen = env.get_current_state_fen()
        
        # Apply e2e4
        env.apply_move("e2e4")
        self.assertNotEqual(env.get_current_state_fen(), initial_fen)
        # FEN after e2e4: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
        expected_fen_after_e2e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        self.assertEqual(env.get_current_state_fen(), expected_fen_after_e2e4)
        self.assertFalse(env.get_current_player(), "Player should be Black after White moves")
        
        # Apply c7c5 (Sicilian Defense)
        env.apply_move("c7c5")
        # FEN after e2e4 c7c5: rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2
        expected_fen_after_c7c5 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        self.assertEqual(env.get_current_state_fen(), expected_fen_after_c7c5)
        self.assertTrue(env.get_current_player(), "Player should be White after Black moves")


    def test_apply_invalid_move_format(self):
        env = ChessEnvironment()
        with self.assertRaisesRegex(ValueError, "Invalid or illegal move: e2e8"):
            env.apply_move("e2e8") # Pawn cannot jump that far

    def test_apply_illegal_move_logic(self):
        env = ChessEnvironment()
        env.apply_move("e2e4") # White moves
        # Trying to move e7e5 (Black's pawn) when it's Black's turn IS legal
        # Trying to move e2e4 again (White's pawn) when it's Black's turn is NOT legal
        with self.assertRaisesRegex(ValueError, "Invalid or illegal move: e2e4"):
            env.apply_move("e2e4")

    def test_game_not_over_at_start(self):
        env = ChessEnvironment()
        self.assertFalse(env.is_game_over())
        self.assertIsNone(env.get_game_outcome())

    def test_checkmate(self):
        # Fool's Mate
        env = ChessEnvironment()
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
        for move in moves:
            env.apply_move(move)
        
        self.assertTrue(env.is_game_over())
        outcome = env.get_game_outcome()
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["winner"], "BLACK")
        self.assertEqual(outcome["reason"], "CHECKMATE")

    def test_stalemate(self):
        # Example of a stalemate position
        # Board: Black King on h8, White Queen on g6, White King on f7. White to move.
        # White moves Qg7, results in stalemate.
        env = ChessEnvironment()
        # Create a custom position leading to stalemate
        # k7/5K1/6Q1/8/8/8/8/8 b - - 0 1 (Black King a8, White King f7, White Queen g6, Black to move)
        # Let's simplify, set up a known stalemate:
        # 8/8/8/8/8/5k2/5q2/7K w - - 0 1 (Black king f3, white queen f2, white king h1. White to move. Qf1# is not possible)
        # A standard stalemate: White: Kh1, Qg2. Black: Kf3. White to move Qg1 -> stalemate.
        # Simplified: K7/8/k1Q5/8/8/8/8/8 w - - Vs K on a6, Q on c6, white to move. Ka5. Stalemate
        # Let's use a known FEN for stalemate setup if possible, or construct one
        # FEN for a position where next move is stalemate: 7k/5Q2/8/8/8/8/8/K7 w - - 0 1 (White to move Qf6 stalemate)
        # Actually, an easier FEN for stalemate (Black is stalemated): k7/P1P1P1P1/8/8/8/8/8/K7 w - - 0 1
        # Even simpler: White: Ka1, Black: Kc3, Pc2. White to move. White has no legal moves. (Not a stalemate by rule but no moves)
        # This is harder to set up quickly without complex FEN strings. Let's use python-chess's ability to make moves to reach one.
        
        # White: Kg1, Qg5. Black: Ke1. White to move Qe3. Black has no moves. Stalemate.
        # FEN: 4k3/8/8/6q1/8/8/8/6K1 w - - 0 1. White to move Kf2. Then Black Qe2 is stalemate.
        # board = chess.Board("4k3/8/8/6q1/8/8/8/6K1 w - - 0 1")
        # board.push_uci("g1f2") # Kf2
        # board.push_uci("e8f8") # Kf8, queen still attacks e1
        # board.push_uci("g5e3") #Qe3
        
        # FEN for a clear stalemate: 5k2/8/8/8/8/8/5Q2/6K1 b - - 0 1 (Black to move, King is not in check, no legal moves)
        # Let's set this FEN directly in the board object for the test for simplicity
        env.board = chess.Board("7k/5Q2/8/8/8/8/K7/8 b - - 0 1") # Black to move, but has no legal moves and is not in check
        self.assertTrue(env.board.is_stalemate()) # Verify our setup
        self.assertTrue(env.is_game_over())
        outcome = env.get_game_outcome()
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["winner"], "DRAW")
        self.assertEqual(outcome["reason"], "STALEMATE")

    def test_insufficient_material(self):
        # King vs King
        env = ChessEnvironment()
        env.board = chess.Board("k7/8/K7/8/8/8/8/8 w - - 0 1") # Only kings left
        self.assertTrue(env.is_game_over()) # Should be game over due to insufficient material
        outcome = env.get_game_outcome()
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["winner"], "DRAW")
        self.assertEqual(outcome["reason"], "INSUFFICIENT_MATERIAL")

    def test_reset_environment(self):
        env = ChessEnvironment()
        # Make a few moves
        env.apply_move("e2e4")
        env.apply_move("e7e5")
        self.assertNotEqual(env.get_current_state_fen(), chess.STARTING_FEN, "Board should have changed.")
        
        env.reset()
        self.assertEqual(env.get_current_state_fen(), chess.STARTING_FEN, "Board should be reset to starting FEN.")
        self.assertTrue(env.get_current_player(), "Player should be White after reset.")
        self.assertEqual(len(env.get_legal_moves()), 20, "Should have 20 legal moves after reset.")

    def test_get_scalar_outcome(self):
        env = ChessEnvironment()
        self.assertIsNone(env.get_scalar_outcome(), "Scalar outcome should be None for ongoing game.")

        # Test White wins (Fool's Mate, Black is checkmated, White was last to move)
        # board.turn will be Black (the one checkmated)
        env.reset()
        # f3 e5 g4 Qh4#
        # 1. f2f3 (White)
        # 2. e7e5 (Black)
        # 3. g2g4 (White)
        # 4. d8h4# (Black wins, White is checkmated, board.turn is White)
        # Let's do a simpler checkmate: White checkmates Black
        # 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
        env.apply_move("e2e4") # W
        env.apply_move("e7e5") # B
        env.apply_move("d1h5") # W
        env.apply_move("b8c6") # B
        env.apply_move("f1c4") # W
        env.apply_move("g8f6") # B
        env.apply_move("h5f7") # W (Qxf7#)
        # Game is over, Black is checkmated, it's Black's turn (but Black has no moves)
        # The winner is White. Player at game end (board.turn) is Black.
        # If White won, and player at game end is Black (who lost), scalar is -1 from Black's perspective.
        # Value should be from perspective of player whose turn it WOULD be.
        # AlphaZero stores (value for the *next* player).
        # Let's define value as: value for the player whose turn it was *just before* the game-ending move.
        # Or more simply: +1 if White won, -1 if Black won, 0 for Draw.
        # Then, the RL agent knows if it's playing as White or Black.
        # The prompt said: "Value Head: Outputs a scalar predicting the game outcome (-1 for loss, O for draw, +1 for win)."
        # This is typically from the perspective of the current player being evaluated.
        # For training data (state, MCTS_policy, game_outcome): game_outcome is the *actual* result for the player who was at that state.

        # Let's redefine get_scalar_outcome to be simpler:
        # 1 if White wins, -1 if Black wins, 0 for Draw.
        # The RL agent can then flip the sign if it's Black. This is common.
        # (Revising the get_scalar_outcome logic slightly for simplicity and common RL practice)

        # --- Revised approach for get_scalar_outcome logic in the main code ---
        # (I will mentally update the main code logic for get_scalar_outcome during this thought)
        # If White wins -> 1
        # If Black wins -> -1
        # If Draw -> 0
        # --- End revised approach ---

        # Test White wins (Scholar's Mate)
        env.reset()
        moves_white_wins = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
        for move in moves_white_wins:
            env.apply_move(move)
        self.assertTrue(env.is_game_over())
        self.assertEqual(env.get_game_outcome()["winner"], "WHITE")
        # Based on revised logic: White wins = 1
        self.assertEqual(env.get_scalar_outcome(), 1)

        # Test Black wins (Fool's Mate)
        env.reset()
        moves_black_wins = ["f2f3", "e7e5", "g2g4", "d8h4"]
        for move in moves_black_wins:
            env.apply_move(move)
        self.assertTrue(env.is_game_over())
        self.assertEqual(env.get_game_outcome()["winner"], "BLACK")
        # Based on revised logic: Black wins = -1
        self.assertEqual(env.get_scalar_outcome(), -1)

        # Test Draw (Stalemate)
        env.reset()
        # Using the stalemate FEN from previous successful test
        env.board = chess.Board("7k/5Q2/8/8/8/8/K7/8 b - - 0 1") # Black to move, stalemated by Qf7
        self.assertTrue(env.is_game_over())
        self.assertEqual(env.get_game_outcome()["winner"], "DRAW")
        self.assertEqual(env.get_scalar_outcome(), 0)

        # Test Draw (Insufficient Material)
        env.reset()
        env.board = chess.Board("k7/8/K7/8/8/8/8/8 w - - 0 1")
        self.assertTrue(env.is_game_over())
        self.assertEqual(env.get_game_outcome()["winner"], "DRAW")
        self.assertEqual(env.get_scalar_outcome(), 0)        


if __name__ == '__main__':
    unittest.main()