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

from ..neural_network.hybrid_rnn_model import HybridRNNModel
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input

class Trainer:
    def __init__(self, model_config: Dict[str, Any], device: torch.device = torch.device("cpu")):
        self.network: Optional[HybridRNNModel] = None
        self.model_config = model_config
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.loss_criterion = MSELoss()
        # --- RE-INTEGRATED: Weight for the material loss term ---
        self.material_balance_loss_weight = self.model_config.get('MATERIAL_BALANCE_LOSS_WEIGHT', 0.5)

    def _initialize_new_network(self) -> Tuple[HybridRNNModel, int]:
        # This method now correctly points to the updated HybridRNNModel constructor
        print("Creating new network from scratch...")
        self.network = HybridRNNModel(
            gnn_hidden_dim=self.model_config.get('GNN_HIDDEN_DIM', 128),
            cnn_in_channels=self.model_config.get('CNN_INPUT_CHANNELS', 14),
            embed_dim=self.model_config.get('EMBED_DIM', 256),
            num_heads=self.model_config.get('NUM_HEADS', 4),
            gnn_metadata=(['square', 'piece'], [('piece', 'occupies', 'square'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece'), ('square', 'adjacent_to', 'square')]),
            rnn_hidden_dim=self.model_config.get('RNN_HIDDEN_DIM', 512),
            num_rnn_layers=self.model_config.get('RNN_NUM_LAYERS', 2),
            policy_size=get_action_space_size()
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.model_config['LEARNING_RATE'], weight_decay=self.model_config['WEIGHT_DECAY'])
        self.scheduler = StepLR(self.optimizer, step_size=self.model_config.get('LR_SCHEDULER_STEP_SIZE', 100), gamma=self.model_config.get('LR_SCHEDULER_GAMMA', 0.9))
        return self.network, 0

    # ... (load_or_initialize_network and other helpers are unchanged from my last response) ...
    def _get_game_number_from_filename(self, filepath: Path) -> int:
        match = re.search(r'_game_(\d+)', filepath.name)
        return int(match.group(1)) if match else -1

    def load_or_initialize_network(self, directory: Optional[Path], specific_checkpoint_path: Optional[Path] = None) -> Tuple[HybridRNNModel, int]:
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
                    if isinstance(v, torch.Tensor): state[k] = v.to(self.device)
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
        
    def train_on_batch(self, game_examples: List[Tuple[str, Dict, float]], 
                       puzzle_examples: List[Dict], batch_size: int, puzzle_ratio: float) -> Tuple[float, float, float]:
        if not game_examples and not puzzle_examples:
            return 0.0, 0.0, 0.0

        self.network.train()
        total_policy_loss, total_value_loss, total_material_loss, batches_processed = 0.0, 0.0, 0.0, 0

        # This training loop processes one game sequence at a time.
        # This is a simplification for now; true batching of sequences can be a future optimization.
        for game in game_examples:
            self.optimizer.zero_grad()
            
            fen_strings, mcts_policies, game_outcomes = zip(*game)
            boards = [chess.Board(fen) for fen in fen_strings]
            
            gnn_data_list, cnn_data_list, material_targets_list = zip(*[convert_to_gnn_input(b, self.device) for b in boards])
            
            policy_targets = torch.stack([self._convert_mcts_policy_to_tensor(p, b, get_action_space_size()) for p, b in zip(mcts_policies, boards)])
            value_targets = torch.tensor(game_outcomes, dtype=torch.float32, device=self.device).view(-1, 1)
            material_targets = torch.stack(material_targets_list)

            hidden_state = torch.zeros((self.network.num_rnn_layers, 1, self.network.rnn_hidden_dim), device=self.device)

            seq_policy_loss, seq_value_loss, seq_material_loss = 0, 0, 0
            
            for i in range(len(boards)):
                gnn_batch = Batch.from_data_list([gnn_data_list[i]])
                cnn_batch = cnn_data_list[i].unsqueeze(0)

                # The model now returns 4 items, with material_balance as the third
                pred_policy_logits, pred_value, pred_material, new_hidden_state = self.network(gnn_batch, cnn_batch, hidden_state)
                
                hidden_state = new_hidden_state.detach()
                
                # --- RE-INTEGRATED: Loss calculations for all 3 heads ---
                seq_policy_loss += F.cross_entropy(pred_policy_logits, policy_targets[i].unsqueeze(0))
                seq_value_loss += self.loss_criterion(pred_value, value_targets[i].unsqueeze(0))
                seq_material_loss += self.loss_criterion(pred_material, material_targets[i].unsqueeze(0))

            avg_policy_loss = seq_policy_loss / len(boards)
            avg_value_loss = seq_value_loss / len(boards)
            avg_material_loss = seq_material_loss / len(boards)

            value_loss_weight = self.model_config.get('VALUE_LOSS_WEIGHT', 1.0)
            total_loss = avg_policy_loss + (avg_value_loss * value_loss_weight) + (avg_material_loss * self.material_balance_loss_weight)
            
            total_loss.backward()
            self.optimizer.step()

            total_policy_loss += avg_policy_loss.item()
            total_value_loss += avg_value_loss.item()
            total_material_loss += avg_material_loss.item()
            batches_processed += 1
            
        # (Puzzle training is omitted for clarity in this step, can be re-added later)

        if self.scheduler:
            self.scheduler.step()

        avg_p_loss = total_policy_loss / batches_processed if batches_processed > 0 else 0
        avg_v_loss = total_value_loss / batches_processed if batches_processed > 0 else 0
        avg_m_loss = total_material_loss / batches_processed if batches_processed > 0 else 0
        
        return avg_p_loss, avg_v_loss, avg_m_loss

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