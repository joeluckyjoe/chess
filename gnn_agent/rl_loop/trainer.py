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

from ..neural_network.hybrid_transformer_model import HybridTransformerModel # MODIFIED: Import new model
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size
from ..gamestate_converters.gnn_data_converter import convert_to_gnn_input

class Trainer:
    def __init__(self, model_config: Dict[str, Any], device: torch.device = torch.device("cpu")):
        self.network: Optional[HybridTransformerModel] = None # MODIFIED: Type hint
        self.model_config = model_config
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.loss_criterion = MSELoss()
        self.material_balance_loss_weight = self.model_config.get('MATERIAL_BALANCE_LOSS_WEIGHT', 0.5)

    def _initialize_new_network(self) -> Tuple[HybridTransformerModel, int]: # MODIFIED: Type hint
        """Instantiates the new HybridTransformerModel."""
        print("Creating new Transformer network from scratch...")
        self.network = HybridTransformerModel(
            gnn_hidden_dim=self.model_config.get('GNN_HIDDEN_DIM', 128),
            cnn_in_channels=self.model_config.get('CNN_INPUT_CHANNELS', 14),
            embed_dim=self.model_config.get('EMBED_DIM', 256),
            gnn_num_heads=self.model_config.get('GNN_NUM_HEADS', 4),
            transformer_nhead=self.model_config.get('TRANSFORMER_NHEAD', 8),
            transformer_nlayers=self.model_config.get('TRANSFORMER_NLAYERS', 4),
            transformer_dim_feedforward=self.model_config.get('TRANSFORMER_DIM_FEEDFORWARD', 512),
            gnn_metadata=(['square', 'piece'], [('piece', 'occupies', 'square'), ('piece', 'attacks', 'piece'), ('piece', 'defends', 'piece'), ('square', 'adjacent_to', 'square')]),
            policy_size=get_action_space_size()
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.model_config['LEARNING_RATE'], weight_decay=self.model_config['WEIGHT_DECAY'])
        self.scheduler = StepLR(self.optimizer, step_size=self.model_config.get('LR_SCHEDULER_STEP_SIZE', 100), gamma=self.model_config.get('LR_SCHEDULER_GAMMA', 0.9))
        return self.network, 0

    def _get_game_number_from_filename(self, filepath: Path) -> int:
        match = re.search(r'_game_(\d+)', filepath.name)
        return int(match.group(1)) if match else -1

    def load_or_initialize_network(self, directory: Optional[Path], specific_checkpoint_path: Optional[Path] = None) -> Tuple[HybridTransformerModel, int]: # MODIFIED: Type hint
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
            # Add strict=False to handle architectural changes gracefully
            self.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
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
        total_policy_loss, total_value_loss, total_material_loss, games_processed = 0.0, 0.0, 0.0, 0

        # --- MAJOR REFACTOR FOR TRANSFORMER ---
        # The training loop now processes one full game sequence at a time
        for game in game_examples:
            self.optimizer.zero_grad()
            
            # 1. Unpack and prepare the entire sequence
            fen_strings, mcts_policies, game_outcomes = zip(*game)
            if not fen_strings: continue # Skip empty games
            
            boards = [chess.Board(fen) for fen in fen_strings]
            
            # 2. Convert all boards in the sequence to GNN/CNN inputs
            conversion_results = [convert_to_gnn_input(b, self.device) for b in boards]
            gnn_data_list, cnn_data_list, material_targets_list = zip(*conversion_results)
            
            # 3. Batch/Stack the data for the entire sequence
            gnn_batch_for_seq = Batch.from_data_list(list(gnn_data_list))
            cnn_tensor_for_seq = torch.stack(list(cnn_data_list))
            
            # 4. Prepare targets for the entire sequence
            policy_targets = torch.stack([self._convert_mcts_policy_to_tensor(p, b, get_action_space_size()) for p, b in zip(mcts_policies, boards)])
            value_targets = torch.tensor(game_outcomes, dtype=torch.float32, device=self.device).view(-1, 1)
            material_targets = torch.stack(list(material_targets_list))

            # 5. Perform a single forward pass for the whole sequence
            # The model now returns predictions for every position in the sequence
            pred_policy_logits, pred_values, pred_materials = self.network(gnn_batch_for_seq, cnn_tensor_for_seq)

            # 6. Calculate loss across the entire sequence
            policy_loss = F.cross_entropy(pred_policy_logits, policy_targets)
            value_loss = self.loss_criterion(pred_values, value_targets)
            material_loss = self.loss_criterion(pred_materials, material_targets)

            value_loss_weight = self.model_config.get('VALUE_LOSS_WEIGHT', 1.0)
            total_loss = policy_loss + (value_loss * value_loss_weight) + (material_loss * self.material_balance_loss_weight)
            
            # 7. Backward pass and optimizer step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.model_config.get('CLIP_GRAD_NORM', 1.0))
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_material_loss += material_loss.item()
            games_processed += 1

        if self.scheduler:
            self.scheduler.step()

        avg_p_loss = total_policy_loss / games_processed if games_processed > 0 else 0
        avg_v_loss = total_value_loss / games_processed if games_processed > 0 else 0
        avg_m_loss = total_material_loss / games_processed if games_processed > 0 else 0
        
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