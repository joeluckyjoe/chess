# gnn_agent/rl_loop/trainer.py
import torch
import os
import re
import random
from datetime import datetime
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from typing import Dict, List, Tuple, Any, Optional
import chess

from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input

class Trainer:
    """
    Manages the training process and checkpointing for the ChessNetwork.
    """
    def __init__(self, model_config: Dict[str, Any], device: torch.device = torch.device("cpu")):
        self.network = None
        self.model_config = model_config
        self.device = device
        self.optimizer = None
        self.scheduler = None 
        self.value_criterion = MSELoss()

    def _initialize_new_network(self) -> Tuple[ChessNetwork, int]:
        """
        CORRECTED: Instantiates the new self-contained ChessNetwork directly,
        passing hyperparameters from the model_config.
        """
        print("Creating new network from scratch...")
        
        self.network = ChessNetwork(
            embed_dim=self.model_config.get('EMBED_DIM', 256),
            gnn_hidden_dim=self.model_config.get('GNN_HIDDEN_DIM', 128),
            num_heads=self.model_config.get('NUM_HEADS', 4)
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.model_config['LEARNING_RATE'], 
            weight_decay=self.model_config['WEIGHT_DECAY']
        )
        
        self.scheduler = StepLR(
            self.optimizer, 
            step_size=self.model_config.get('LR_SCHEDULER_STEP_SIZE', 100), 
            gamma=self.model_config.get('LR_SCHEDULER_GAMMA', 0.9)
        )
        return self.network, 0

    def _get_game_number_from_filename(self, filepath: Path) -> int:
        match = re.search(r'_game_(\d+)', filepath.name)
        return int(match.group(1)) if match else -1

    def load_or_initialize_network(self, directory: Optional[Path], specific_checkpoint_path: Optional[Path] = None) -> Tuple[ChessNetwork, int]:
        file_to_load = None
        if specific_checkpoint_path and specific_checkpoint_path.exists():
            file_to_load = specific_checkpoint_path
        elif directory and directory.is_dir():
            files = [f for f in directory.glob('*.pth.tar')]
            if files:
                file_to_load = max(files, key=self._get_game_number_from_filename)

        if not file_to_load:
            print("No checkpoint found or specified. Initializing new network.")
            return self._initialize_new_network()

        # Initialize network structure FIRST before attempting to load state dict
        # This creates the optimizer and scheduler as well.
        self._initialize_new_network() 
        
        try:
            print(f"Loading checkpoint: {file_to_load}")
            checkpoint = torch.load(file_to_load, map_location=self.device)
            
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state loaded from checkpoint.")

            game_number = checkpoint.get('game_number', 0)
            
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
                        
            print(f"Checkpoint loaded successfully. Resuming from game {game_number + 1}.")
            return self.network, game_number
        except Exception as e:
            print(f"Error loading checkpoint {file_to_load}: {e}. Initializing new network.")
            # If loading fails, we've already initialized a fresh network.
            return self._initialize_new_network()

    def _convert_mcts_policy_to_tensor(self, mcts_policy_dict: Dict[chess.Move, float], board: chess.Board, action_space_size: int) -> torch.Tensor:
        policy_tensor = torch.zeros(action_space_size, device=self.device)
        if not mcts_policy_dict: return policy_tensor
        for move, prob in mcts_policy_dict.items():
            try:
                idx = move_to_index(move, board)
                policy_tensor[idx] = prob
            except Exception:
                pass
        return policy_tensor

    def _convert_puzzles_to_training_format(self, puzzle_examples: List[Dict]) -> List[Tuple[str, Dict[chess.Move, float], float, str]]:
        converted_puzzles = []
        for puzzle in puzzle_examples:
            try:
                board = chess.Board(puzzle['fen'])
                best_move = chess.Move.from_uci(puzzle['best_move_uci'])
                policy_dict = {best_move: 1.0}
                converted_puzzles.append((board.fen(), policy_dict, 1.0, 'puzzle'))
            except Exception:
                pass
        return converted_puzzles

    def train_on_batch(self, game_examples: List[Tuple[str, Dict[chess.Move, float], float]], 
                       puzzle_examples: List[Dict], batch_size: int, puzzle_ratio: float = 0.25):
        if not game_examples and not puzzle_examples:
            return 0.0, 0.0

        converted_puzzles = self._convert_puzzles_to_training_format(puzzle_examples)
        tagged_game_examples = [(fen, policy, outcome, 'game') for fen, policy, outcome in game_examples]
        
        num_puzzles_to_add = int(len(tagged_game_examples) * puzzle_ratio)
        sampled_puzzles = []
        if converted_puzzles:
            sampled_puzzles = random.sample(converted_puzzles, min(num_puzzles_to_add, len(converted_puzzles)))

        all_data = tagged_game_examples + sampled_puzzles
        random.shuffle(all_data)

        self.network.train()
        total_policy_loss, total_value_loss, game_data_count = 0.0, 0.0, 0
        
        for i in range(0, len(all_data), batch_size):
            self.optimizer.zero_grad()
            batch_chunk = all_data[i:i+batch_size]
            if not batch_chunk: continue
            
            fen_strings, mcts_policies, game_outcomes, data_types = zip(*batch_chunk)
            
            boards = [chess.Board(fen) for fen in fen_strings]
            data_list = [convert_to_gnn_input(b, torch.device('cpu')) for b in boards]

            # Manually collate all tensors for the dual-graph structure
            square_features = torch.cat([d.square_features for d in data_list], dim=0).to(self.device)
            csum_sq = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor([s.size(0) for s in data_list]), 0)[:-1]])
            square_edge_index = torch.cat([d.square_edge_index + c for d, c in zip(data_list, csum_sq)], dim=1).to(self.device)
            square_batch = torch.tensor([i for i, d in enumerate(data_list) for _ in range(d.square_features.size(0))], dtype=torch.long).to(self.device)
            
            piece_features = torch.cat([d.piece_features for d in data_list], dim=0).to(self.device)
            csum_pc = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor([p.size(0) for p in data_list]), 0)[:-1]])
            piece_edge_index = torch.cat([d.piece_edge_index + c for d, c in zip(data_list, csum_pc)], dim=1).to(self.device)
            piece_batch = torch.tensor([i for i, d in enumerate(data_list) for _ in range(d.piece_features.size(0))], dtype=torch.long).to(self.device)
            piece_to_square_map = torch.cat([d.piece_to_square_map + c for d, c in zip(data_list, csum_sq)], dim=0).to(self.device)
            
            max_pieces = max(d.piece_features.size(0) for d in data_list) if data_list else 0
            piece_padding_mask = torch.ones((len(batch_chunk), max_pieces), dtype=torch.bool, device=self.device)
            for j, d in enumerate(data_list):
                if d.piece_features.size(0) > 0: piece_padding_mask[j, :d.piece_features.size(0)] = 0

            policy_targets = torch.stack([self._convert_mcts_policy_to_tensor(p, b, get_action_space_size()) for p, b in zip(mcts_policies, boards)]).to(self.device)
            value_targets = torch.tensor(game_outcomes, dtype=torch.float32, device=self.device).view(-1, 1)
            
            pred_policy_logits, pred_value = self.network(
                square_features, square_edge_index, square_batch,
                piece_features, piece_edge_index, piece_batch,
                piece_to_square_map, piece_padding_mask
            )

            policy_loss = F.cross_entropy(pred_policy_logits, policy_targets)
            is_game_data_mask = torch.tensor([t == 'game' for t in data_types], dtype=torch.bool, device=self.device)
            
            value_loss = torch.tensor(0.0, device=self.device)
            if is_game_data_mask.any():
                value_loss = self.value_criterion(pred_value[is_game_data_mask], value_targets[is_game_data_mask])
                total_value_loss += value_loss.item() * is_game_data_mask.sum().item()
                game_data_count += is_game_data_mask.sum().item()

            total_policy_loss += policy_loss.item() * len(batch_chunk)
            total_loss = policy_loss + value_loss
            
            if total_loss > 0:
                total_loss.backward()
                self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        num_samples = len(all_data)
        avg_policy_loss = total_policy_loss / num_samples if num_samples > 0 else 0
        avg_value_loss = total_value_loss / game_data_count if game_data_count > 0 else 0
        
        return avg_policy_loss, avg_value_loss
    
    def save_checkpoint(self, directory: Path, game_number: int, filename_override: Optional[str] = None):
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
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config_params': self.model_config,
        }
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")