# FILENAME: gnn_agent/rl_loop/trainer.py
import torch
import os
import re
import random
from datetime import datetime
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
from typing import Dict, List, Tuple, Any, Optional

import chess
from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input
from torch_geometric.data import Batch

class Trainer:
    """
    Manages the training process and checkpointing for the ChessNetwork.
    """
    def __init__(self, model_config: Dict[str, Any], network: ChessNetwork = None, learning_rate: float = 0.001, weight_decay: float = 0.0, device: torch.device = torch.device("cpu")):
        self.network = network
        self.model_config = model_config
        self.device = device
        self.optimizer = None
        self.value_criterion = MSELoss()
        
        if self.network:
            self.network.to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _initialize_new_network(self) -> Tuple[ChessNetwork, int]:
        from ..neural_network.gnn_models import SquareGNN, PieceGNN
        from ..neural_network.attention_module import CrossAttentionModule
        from ..neural_network.policy_value_heads import PolicyHead, ValueHead

        print("Creating new network from scratch...")
        square_gnn = SquareGNN(in_features=12, hidden_features=256, out_features=128, heads=4)
        piece_gnn = PieceGNN(in_channels=12, hidden_channels=256, out_channels=128)
        cross_attention = CrossAttentionModule(sq_embed_dim=128, pc_embed_dim=128, num_heads=4)
        policy_head = PolicyHead(embedding_dim=128, num_possible_moves=get_action_space_size())
        value_head = ValueHead(embedding_dim=128)

        self.network = ChessNetwork(
            square_gnn=square_gnn,
            piece_gnn=piece_gnn,
            cross_attention=cross_attention,
            policy_head=policy_head,
            value_head=value_head
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.model_config['LEARNING_RATE'], 
            weight_decay=self.model_config['WEIGHT_DECAY']
        )
        return self.network, 0

    def _get_game_number_from_filename(self, filepath: Path) -> int:
        match = re.search(r'_game_(\d+)', filepath.name)
        if match:
            return int(match.group(1))
        return -1

    def load_or_initialize_network(self, directory: Path, specific_checkpoint_path: Optional[Path] = None) -> Tuple[ChessNetwork, int]:
        file_to_load = None
        if specific_checkpoint_path and specific_checkpoint_path.exists():
            file_to_load = specific_checkpoint_path
        elif not specific_checkpoint_path:
            if directory.is_dir():
                files = [f for f in directory.glob('*.pth.tar')]
                if files:
                    file_to_load = max(files, key=self._get_game_number_from_filename)

        if not file_to_load:
            print("No checkpoint found or specified. Initializing new network.")
            return self._initialize_new_network()

        try:
            print(f"Loading checkpoint: {file_to_load}")
            # Initialize network structure first before loading state dict
            self._initialize_new_network()
            checkpoint = torch.load(file_to_load, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            game_number = checkpoint.get('game_number', 0)
            # Ensure optimizer state is on the correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print(f"Checkpoint loaded successfully. Resuming from game {game_number + 1}.")
            return self.network, game_number
        except Exception as e:
            print(f"Error loading checkpoint {file_to_load}: {e}. Initializing new network.")
            return self._initialize_new_network()

    def _convert_mcts_policy_to_tensor(self, mcts_policy_dict: Dict[chess.Move, float], board: chess.Board, action_space_size: int) -> torch.Tensor:
        """Converts an MCTS policy dictionary to a dense tensor, now with board context."""
        policy_tensor = torch.zeros(action_space_size, device=self.device)
        if not mcts_policy_dict:
            return policy_tensor
        for move, prob in mcts_policy_dict.items():
            try:
                # --- Pass the board object to move_to_index ---
                idx = move_to_index(move, board)
                policy_tensor[idx] = prob
            except Exception as e:
                # This can happen if a move from an old action space is loaded.
                # It's safe to ignore as it won't contribute to the policy target.
                pass
        return policy_tensor

    def _convert_puzzles_to_training_format(self, puzzle_examples: List[Dict]) -> List[Tuple[str, Dict[chess.Move, float], float, str]]:
        """
        Converts raw puzzle data into the standard training format.
        Now returns a 4-tuple that includes a 'type' tag.
        """
        converted_puzzles = []
        for puzzle in puzzle_examples:
            try:
                board = chess.Board(puzzle['fen'])
                # The puzzle is from the perspective of the side to move. A correct move leads to a win.
                outcome = 1.0 # This outcome is for policy training only and should be ignored for value training.
                # The policy is deterministic: 100% probability on the single best move.
                best_move = chess.Move.from_uci(puzzle['best_move_uci'])
                policy_dict = {best_move: 1.0}
                # Add the 'puzzle' tag
                converted_puzzles.append((board.fen(), policy_dict, outcome, 'puzzle'))
            except Exception as e:
                pass
        return converted_puzzles

    def train_on_batch(self, game_examples: List[Tuple[str, Dict[chess.Move, float], float]], 
                       puzzle_examples: List[Dict], batch_size: int, puzzle_ratio: float = 0.25):
        """
        Performs a single training step on a mixed batch of game and puzzle data.
        This version is rewritten to use PyTorch Geometric's batching for efficiency.
        """
        if not game_examples and not puzzle_examples:
            return 0.0, 0.0

        converted_puzzles = self._convert_puzzles_to_training_format(puzzle_examples)
        tagged_game_examples = [(fen, policy, outcome, 'game') for fen, policy, outcome in game_examples]
        
        num_puzzles_to_add = int(len(tagged_game_examples) * puzzle_ratio)
        sampled_puzzles = []
        if len(converted_puzzles) > 0:
            sampled_puzzles = random.sample(converted_puzzles, min(num_puzzles_to_add, len(converted_puzzles)))

        all_data = tagged_game_examples + sampled_puzzles
        random.shuffle(all_data)

        self.network.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        game_data_count = 0
        
        for i in range(0, len(all_data), batch_size):
            self.optimizer.zero_grad()
            
            batch_chunk = all_data[i:i+batch_size]
            if not batch_chunk:
                continue
            
            fen_strings, mcts_policies, game_outcomes, data_types = zip(*batch_chunk)

            # --- PHASE AB REFACTOR: BATCHED DATA PREPARATION ---
            # 1. Convert all boards in the chunk to PyG Data objects and collate them.
            #    This assumes `convert_to_gnn_input` returns a PyG Data object.
            boards = [chess.Board(fen) for fen in fen_strings]
            data_list = [convert_to_gnn_input(b, self.device) for b in boards]
            
            # `Batch.from_data_list` creates a single large graph object.
            # It handles concatenating features and creating the `batch` index tensor.
            data_batch = Batch.from_data_list(data_list)

            # 2. Prepare policy and value targets for the entire batch.
            action_space_size = get_action_space_size()
            policy_targets = torch.stack([
                self._convert_mcts_policy_to_tensor(p, b, action_space_size) 
                for p, b in zip(mcts_policies, boards)
            ]).to(self.device)
            value_targets = torch.tensor(game_outcomes, dtype=torch.float32, device=self.device).view(-1, 1)
            
            # 3. Perform a single forward pass with the entire batch.
            #    The network's forward method now receives all required arguments from the Batch object.
            #    Note: The padding mask is not needed here as we are not doing MCTS-style search.
            pred_policy_logits, pred_value = self.network(
                square_features=data_batch.square_features,
                square_edge_index=data_batch.square_edge_index,
                square_batch=data_batch.square_features_batch,
                piece_features=data_batch.piece_features,
                piece_edge_index=data_batch.piece_edge_index,
                piece_batch=data_batch.piece_features_batch,
                piece_to_square_map=data_batch.piece_to_square_map,
                piece_padding_mask=None 
            )

            # 4. Calculate loss for the entire batch.
            policy_loss = F.cross_entropy(pred_policy_logits, policy_targets)
            
            # Create a boolean mask to filter for 'game' data for value loss calculation.
            is_game_data_mask = torch.tensor([t == 'game' for t in data_types], dtype=torch.bool, device=self.device)
            
            value_loss = 0.0
            if is_game_data_mask.any():
                # Filter predictions and targets using the mask before calculating loss.
                value_loss = self.value_criterion(pred_value[is_game_data_mask], value_targets[is_game_data_mask])
                total_value_loss += value_loss.item() * is_game_data_mask.sum()
                game_data_count += is_game_data_mask.sum().item()

            total_policy_loss += policy_loss.item() * len(batch_chunk)
            
            # The total loss for backpropagation is the policy loss plus the (possibly zero) value loss.
            total_loss = policy_loss + value_loss
            
            if total_loss > 0:
                total_loss.backward()
                self.optimizer.step()
            # --- END REFACTOR ---

        num_samples = len(all_data)
        avg_policy_loss = total_policy_loss / num_samples if num_samples > 0 else 0
        avg_value_loss = total_value_loss / game_data_count if game_data_count > 0 else 0
        
        return avg_policy_loss, avg_value_loss
    
    def save_checkpoint(self, directory: Path, game_number: int, filename_override: Optional[str] = None):
        """Saves the model and optimizer state. Can use a specific filename."""
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)

        if filename_override:
            filename = filename_override
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_game_{game_number}_{timestamp}.pth.tar"
        
        filepath = directory / filename
        state = {
            'game_number': game_number,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config_params': self.model_config,
        }
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")
