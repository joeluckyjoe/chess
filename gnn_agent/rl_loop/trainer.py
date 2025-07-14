# gnn_agent/rl_loop/trainer.py (Updated with Value Loss Weight)
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
from torch_geometric.loader import DataLoader
from typing import Dict, List, Tuple, Any, Optional
import chess

from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input

class Trainer:
    # __init__, _initialize_new_network, and other helpers are unchanged
    def __init__(self, model_config: Dict[str, Any], device: torch.device = torch.device("cpu")):
        self.network = None
        self.model_config = model_config
        self.device = device
        self.optimizer = None
        self.scheduler = None 
        self.value_criterion = MSELoss()

    def _initialize_new_network(self) -> Tuple[ChessNetwork, int]:
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
            return self._initialize_new_network()
        self._initialize_new_network() 
        try:
            checkpoint = torch.load(file_to_load, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            game_number = checkpoint.get('game_number', 0)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            return self.network, game_number
        except Exception as e:
            print(f"Error loading checkpoint {file_to_load}: {e}. Initializing new network.")
            return self._initialize_new_network()

    def _convert_mcts_policy_to_tensor(self, mcts_policy_dict: Dict[chess.Move, float], board: chess.Board, action_space_size: int) -> torch.Tensor:
        policy_tensor = torch.zeros(action_space_size, device=self.device)
        if not mcts_policy_dict: return policy_tensor
        for move, prob in mcts_policy_dict.items():
            try:
                idx = move_to_index(move, board)
                policy_tensor[idx] = prob
            except Exception: pass
        return policy_tensor

    def _convert_puzzles_to_training_format(self, puzzle_examples: List[Dict]) -> List[Tuple[str, Dict[chess.Move, float], float, str]]:
        converted_puzzles = []
        for puzzle in puzzle_examples:
            try:
                board = chess.Board(puzzle['fen'])
                best_move = chess.Move.from_uci(puzzle['best_move_uci'])
                policy_dict = {best_move: 1.0}
                converted_puzzles.append((board.fen(), policy_dict, 1.0, 'puzzle'))
            except Exception: pass
        return converted_puzzles

    def train_on_batch(self, game_examples: List[Tuple[str, Dict[chess.Move, float], float]], 
                       puzzle_examples: List[Dict], batch_size: int, puzzle_ratio: float = 0.25):
        if not game_examples and not puzzle_examples:
            return 0.0, 0.0

        converted_puzzles = self._convert_puzzles_to_training_format(puzzle_examples)
        tagged_game_examples = [(fen, policy, outcome, 'game') for fen, policy, outcome in game_examples]
        num_puzzles_to_add = int(len(tagged_game_examples) * puzzle_ratio)
        sampled_puzzles = random.sample(converted_puzzles, min(num_puzzles_to_add, len(converted_puzzles))) if converted_puzzles else []
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
            
            # --- MODIFICATION FOR GNN+CNN HYBRID MODEL ---
            # convert_to_gnn_input now returns a tuple (gnn_data, cnn_data)
            # We need to handle both inputs separately.
            gnn_data_list, cnn_data_list = zip(*[convert_to_gnn_input(b, torch.device('cpu')) for b in boards])

            # Batch the GNN data using PyG's DataLoader
            gnn_loader = DataLoader(list(gnn_data_list), batch_size=len(gnn_data_list))
            batched_gnn_data = next(iter(gnn_loader))
            batched_gnn_data.to(self.device)

            # Stack the CNN data into a single tensor
            batched_cnn_data = torch.stack(cnn_data_list, 0).to(self.device)
            # --- END MODIFICATION ---
            
            policy_targets = torch.stack([self._convert_mcts_policy_to_tensor(p, b, get_action_space_size()) for p, b in zip(mcts_policies, boards)]).to(self.device)
            value_targets = torch.tensor(game_outcomes, dtype=torch.float32, device=self.device).view(-1, 1)
            
            # --- MODIFICATION FOR GNN+CNN HYBRID MODEL ---
            # Pass both batched GNN and CNN data to the network's forward method
            pred_policy_logits, pred_value = self.network(batched_gnn_data, batched_cnn_data)
            # --- END MODIFICATION ---

            policy_loss = F.cross_entropy(pred_policy_logits, policy_targets)
            is_game_data_mask = torch.tensor([t == 'game' for t in data_types], dtype=torch.bool, device=self.device)
            
            value_loss = torch.tensor(0.0, device=self.device)
            if is_game_data_mask.any():
                value_loss = self.value_criterion(pred_value[is_game_data_mask], value_targets[is_game_data_mask])
                total_value_loss += value_loss.item() * is_game_data_mask.sum().item()
                game_data_count += is_game_data_mask.sum().item()

            total_policy_loss += policy_loss.item() * len(batch_chunk)
            
            value_loss_weight = self.model_config.get('VALUE_LOSS_WEIGHT', 1.0)
            total_loss = policy_loss + (value_loss * value_loss_weight)
            
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
        filename = filename_override or f"checkpoint_game_{game_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth.tar"
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