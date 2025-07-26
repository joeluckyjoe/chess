# FILENAME: /home/giuseppe/chess/gnn_agent/training/trainer.py
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
from torch_geometric.data import Batch
from typing import Dict, List, Tuple, Any, Optional
import chess

# --- MODIFIED: Import the new model ---
from ..neural_network.value_next_state_model import ValueNextStateModel
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


class Trainer:
    def __init__(self, model_config: Dict[str, Any], device):
        # --- MODIFIED: Update type hint and remove material loss weight ---
        self.network: Optional[ValueNextStateModel] = None
        self.model_config = model_config
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.loss_criterion = MSELoss()
        # --- ADDED: Weight for the new loss term ---
        self.next_state_loss_weight = self.model_config.get('NEXT_STATE_LOSS_WEIGHT', 1.0)

    def _initialize_new_network(self) -> Tuple[ValueNextStateModel, int]:
        # --- MODIFIED: Instantiate ValueNextStateModel ---
        print("Creating new ValueNextStateModel from scratch...")
        self.network = ValueNextStateModel(
            gnn_hidden_dim=self.model_config.get('GNN_HIDDEN_DIM', 128),
            cnn_in_channels=self.model_config.get('CNN_INPUT_CHANNELS', 14),
            embed_dim=self.model_config.get('EMBED_DIM', 256),
            gnn_num_heads=self.model_config.get('GNN_NUM_HEADS', 4),
            gnn_metadata=(['piece', 'square'], [('piece', 'occupies', 'square'), ('square', 'rev_occupies', 'piece'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece')]),
            policy_size=get_action_space_size()
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.model_config['LEARNING_RATE'], weight_decay=self.model_config['WEIGHT_DECAY'])
        self.scheduler = StepLR(self.optimizer, step_size=self.model_config.get('LR_SCHEDULER_STEP_SIZE', 100), gamma=self.model_config.get('LR_SCHEDULER_GAMMA', 0.9))
        return self.network, 0

    def _get_game_number_from_filename(self, filepath: Path) -> int:
        match = re.search(r'_game_(\d+)', filepath.name)
        return int(match.group(1)) if match else -1

    def load_or_initialize_network(self, directory: Optional[Path], specific_checkpoint_path: Optional[Path] = None) -> Tuple[ValueNextStateModel, int]:
        # --- MODIFIED: Update return type hint ---
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
            print(f"Loading checkpoint from: {file_to_load}")
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
    
    def _convert_puzzle_to_tensors(self, puzzle_examples: List[Dict]) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
        gnn_data_list, cnn_data_list, policy_targets_list = [], [], []
        for puzzle in puzzle_examples:
            if 'fen' not in puzzle or 'move' not in puzzle:
                print(f"[WARNING] Skipping malformed puzzle: {puzzle}")
                continue
            
            board = chess.Board(puzzle['fen'])
            move = chess.Move.from_uci(puzzle['move'])
            
            gnn_data, cnn_data, _ = convert_to_gnn_input(board, self.device)
            gnn_data_list.append(gnn_data)
            cnn_data_list.append(cnn_data)
            
            policy_target_index = move_to_index(move, board)
            policy_targets_list.append(policy_target_index)
            
        if not gnn_data_list:
            return None, None, None

        return Batch.from_data_list(gnn_data_list), torch.stack(cnn_data_list), torch.tensor(policy_targets_list, dtype=torch.long, device=self.device)

    def train_on_batch(self, game_examples: List[List[Tuple[str, Dict, float, float]]],
                       puzzle_examples: List[Dict], batch_size: int) -> Tuple[float, float, float]:
        
        if not game_examples and not puzzle_examples:
            return 0.0, 0.0, 0.0

        self.network.train()
        total_policy_loss, total_value_loss, total_next_state_loss = 0.0, 0.0, 0.0
        game_batches_processed, puzzle_batches_processed = 0, 0
        
        if puzzle_examples:
            print(f"  -> Training on {len(puzzle_examples)} puzzle examples...")
            num_puzzles = len(puzzle_examples)
            puzzle_indices = list(range(num_puzzles))
            random.shuffle(puzzle_indices)
            
            for i in range(0, num_puzzles, batch_size):
                self.optimizer.zero_grad()
                batch_indices = puzzle_indices[i:i+batch_size]
                batch_puzzles = [puzzle_examples[j] for j in batch_indices]
                
                gnn_batch, cnn_batch, policy_targets = self._convert_puzzle_to_tensors(batch_puzzles)
                
                if gnn_batch is None:
                    continue

                pred_policy_logits, _, _ = self.network(gnn_batch, cnn_batch)
                
                loss = F.cross_entropy(pred_policy_logits, policy_targets)
                loss.backward()
                
                if XLA_AVAILABLE and 'xla' in self.device.type:
                    xm.optimizer_step(self.optimizer)
                else:
                    self.optimizer.step()
                
                total_policy_loss += loss.item()
                puzzle_batches_processed += 1

        for game in game_examples:
            if not game: continue

            print(f"  -> Training on {len(game)} game examples...")

            self.optimizer.zero_grad()
            
            fen_strings, mcts_policies, game_outcomes, next_state_values = zip(*game)
            boards = [chess.Board(fen) for fen in fen_strings]
            
            conversion_results = [convert_to_gnn_input(b, self.device) for b in boards]
            gnn_data_list, cnn_data_list, _ = zip(*conversion_results)
            
            gnn_batch = Batch.from_data_list(list(gnn_data_list))
            cnn_batch = torch.stack(list(cnn_data_list))
            
            policy_targets = torch.stack([self._convert_mcts_policy_to_tensor(p, b, get_action_space_size()) for p, b in zip(mcts_policies, boards)])
            value_targets = torch.tensor(game_outcomes, dtype=torch.float32, device=self.device).view(-1, 1)
            next_state_value_targets = torch.tensor(next_state_values, dtype=torch.float32, device=self.device).view(-1, 1)

            pred_policy_logits, pred_values, pred_next_state_values = self.network(gnn_batch, cnn_batch)

            # --- MODIFIED: This line is corrected to fix the RuntimeError ---
            policy_loss = -torch.sum(policy_targets * F.log_softmax(pred_policy_logits, dim=1), dim=1).mean()
            value_loss = self.loss_criterion(pred_values, value_targets)
            next_state_value_loss = self.loss_criterion(pred_next_state_values, next_state_value_targets)

            value_loss_weight = self.model_config.get('VALUE_LOSS_WEIGHT', 1.0)
            total_loss = policy_loss + (value_loss * value_loss_weight) + (next_state_value_loss * self.next_state_loss_weight)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.model_config.get('CLIP_GRAD_NORM', 1.0))
            
            if XLA_AVAILABLE and 'xla' in self.device.type:
                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_next_state_loss += next_state_value_loss.item()
            game_batches_processed += 1

        if self.scheduler:
            self.scheduler.step()

        total_batches = game_batches_processed + puzzle_batches_processed
        avg_p_loss = total_policy_loss / total_batches if total_batches > 0 else 0
        avg_v_loss = total_value_loss / total_batches if total_batches > 0 else 0
        avg_ns_loss = total_next_state_loss / total_batches if total_batches > 0 else 0
        
        return avg_p_loss, avg_v_loss, avg_ns_loss

    def save_checkpoint(self, directory: Path, game_number: int, filename_override: Optional[str] = None):
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
        filename = filename_override or f"checkpoint_game_{game_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth.tar"
        filepath = directory / filename
        
        save_device = 'cpu'
        self.network.to(save_device)

        state = {
            'game_number': game_number,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config_params': self.model_config,
        }
        
        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")
        
        self.network.to(self.device)