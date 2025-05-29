import unittest
from chess_environment import ChessEnvironment # Adjust import as per your project structure
import chess
import os

# Path to your Stockfish executable
STOCKFISH_PATH = "/usr/games/stockfish" # User provided path
# STOCKFISH_PATH = os.getenv("STOCKFISH_EXECUTABLE_PATH", STOCKFISH_PATH) # For flexibility

# Skip all engine tests if Stockfish path is not valid
skip_engine_tests = not (STOCKFISH_PATH and os.path.exists(STOCKFISH_PATH))
reason_for_skipping = f"Stockfish executable not found at '{STOCKFISH_PATH}' or path not set."

@unittest.skipIf(skip_engine_tests, reason_for_skipping)
class TestChessEnvironmentWithEngine(unittest.TestCase):
    env = None # Class variable for the environment

    @classmethod
    def setUpClass(cls):
        # This check is redundant due to @unittest.skipIf, but good for clarity
        if skip_engine_tests:
            raise unittest.SkipTest(reason_for_skipping)
        try:
            cls.env = ChessEnvironment(uci_engine_path=STOCKFISH_PATH)
        except Exception as e:
            # This makes sure that if __init__ itself fails, the class setup is aborted.
            cls.env = None # Ensure env is None if setup fails
            raise unittest.SkipTest(f"ChessEnvironment initialization failed in setUpClass: {e}")

    @classmethod
    def tearDownClass(cls):
        if cls.env and hasattr(cls.env, 'engine') and cls.env.engine:
            cls.env.close_engine()

    def setUp(self):
        if not self.env:
            # This will skip individual tests if setUpClass failed to set up self.env
            self.skipTest("ChessEnvironment with engine was not initialized in setUpClass.")
        self.env.reset() # Reset board to starting position before each test

    def test_01_engine_initialization_and_responsiveness(self):
        self.assertIsNotNone(self.env.engine, "Engine should be initialized.")
        try:
            # A basic check: analyze the starting position for 1 node
            self.env.engine.analyse(self.env.board, chess.engine.Limit(nodes=1))
        except chess.engine.EngineTerminatedError:
            self.fail("Engine terminated during basic responsiveness test.")
        except chess.engine.EngineError as e:
            self.fail(f"Engine error during basic responsiveness test: {e}")

    def test_02_get_legal_moves_start_pos_from_engine(self):
        legal_moves = self.env.get_legal_moves() # Now uses the engine
        self.assertIsInstance(legal_moves, list, "Legal moves should be a list.")
        self.assertTrue(all(isinstance(m, str) for m in legal_moves), "All moves should be UCI strings.")
        
        expected_moves_at_start = 20
        self.assertEqual(len(legal_moves), expected_moves_at_start,
                         f"Expected {expected_moves_at_start} legal moves at start, got {len(legal_moves)}. Moves: {sorted(legal_moves)}")
        
        self.assertIn("e2e4", legal_moves)
        self.assertIn("g1f3", legal_moves)

    def test_03_get_legal_moves_custom_fen_from_engine(self):
        fen_checkmated_black = "r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 3" # Black is checkmated
        self.env.board = chess.Board(fen_checkmated_black)
        
        legal_moves = self.env.get_legal_moves()
        self.assertEqual(len(legal_moves), 0,
                         f"Expected 0 legal moves for Black in FEN {fen_checkmated_black}, got {len(legal_moves)}. Moves: {legal_moves}")

        fen_few_moves = "k7/8/P7/8/8/8/8/K7 b - - 0 1" # Black king (a8) can only move to b8
        self.env.board = chess.Board(fen_few_moves)
        
        legal_moves = self.env.get_legal_moves()
        self.assertEqual(len(legal_moves), 1, 
                         f"Expected 1 legal move in FEN {fen_few_moves}, got {len(legal_moves)}. Moves: {legal_moves}")
        if len(legal_moves) == 1:
            self.assertIn("a8b8", legal_moves)

    def test_04_pinned_piece_legal_moves_from_engine(self):
        fen_pinned_pawn = "k7/8/8/3q4/8/8/3P4/3K4 w - - 0 1" # White pawn d2 pinned [cite: 111]
        self.env.board = chess.Board(fen_pinned_pawn)
        
        engine_moves = self.env.get_legal_moves()
        print(f"Pinned pawn test ({fen_pinned_pawn}). Engine legal moves: {sorted(engine_moves)}")
        
        self.assertNotIn("d2d3", engine_moves, "Pinned pawn d2 should not be able to move to d3.")
        self.assertNotIn("d2d4", engine_moves, "Pinned pawn d2 should not be able to move to d4.")
        
        self.assertIn("d1c1", engine_moves, "King should be able to move d1c1.")
        self.assertIn("d1e1", engine_moves, "King should be able to move d1e1.")
        self.assertEqual(len(engine_moves), 2, f"Expected exactly 2 legal moves (d1c1, d1e1). Got: {sorted(engine_moves)}")

    def test_05_apply_move_updates_board_state(self):
        initial_fen = self.env.get_current_state_fen()
        self.assertTrue(self.env.get_current_player(), "Should be White's turn initially.") # White's turn
        
        legal_moves = self.env.get_legal_moves()
        self.assertTrue(len(legal_moves) > 0, "Should have legal moves at the start.")
        
        move_to_apply = "e2e4" # A common, known legal move
        if move_to_apply not in legal_moves: # Fallback if e2e4 isn't in the 'd' list for some reason
            move_to_apply = legal_moves[0]
            
        self.env.apply_move(move_to_apply)
        
        new_fen = self.env.get_current_state_fen()
        self.assertNotEqual(initial_fen, new_fen, "FEN should change after applying a move.")
        self.assertFalse(self.env.get_current_player(), "After White moves, it should be Black's turn.") # Black's turn

    def test_06_game_status_methods_after_sequence(self):
        self.env.reset() # Start from initial position

        # Fool's Mate sequence: 1. f3 e5 2. g4 Qh4#
        moves_sequence = ["f2f3", "e7e5", "g2g4", "d8h4"]
        
        for move_uci in moves_sequence:
            # It's good practice to check if the move is in the engine's list first,
            # but for a known sequence, we can try applying directly for this test.
            # Ensure `apply_move` is robust enough or that `get_legal_moves` is called implicitly or explicitly.
            # Our `apply_move` currently relies on `python-chess`'s `push_uci`.
            # For this test, we assume the sequence is valid.
            self.env.apply_move(move_uci)

        self.assertTrue(self.env.is_game_over(), "Game should be over after Fool's Mate.")
        outcome = self.env.get_game_outcome()
        self.assertIsNotNone(outcome, "Outcome should not be None for a completed game.")
        if outcome: 
            self.assertEqual(outcome["winner"], "BLACK", "Black should be the winner in Fool's Mate.")
            self.assertEqual(outcome["reason"], "CHECKMATE", "Reason should be CHECKMATE.")
        self.assertEqual(self.env.get_scalar_outcome(), -1, "Scalar outcome should be -1 for Black win.")

if __name__ == '__main__':
    unittest.main()