import torch
import os
from datetime import datetime
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
from typing import Dict, List, Tuple, Any, Optional
import chess

from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size

class Trainer:
    """
    Manages the training process for the ChessNetwork.
    """
    # --- MODIFIED: Added weight_decay to __init__ and the optimizer ---
    def __init__(self, network: ChessNetwork, model_config: Dict[str, Any], learning_rate: float = 0.001, weight_decay: float = 0.0, device: torch.device = torch.device("cpu")):
        """
        Initializes the Trainer.

        Args:
            network (ChessNetwork): The neural network model to be trained.
            model_config (Dict[str, Any]): Dictionary containing the model's architectural parameters.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer.
            device (torch.device): The device to run training on (cpu or cuda).
        """
        self.network = network
        self.model_config = model_config
        self.device = device
        self.network.to(self.device)
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.value_criterion = MSELoss()

    def _convert_mcts_policy_to_tensor(self, mcts_policy_dict: Dict[chess.Move, float],
                                       action_space_size: int) -> torch.Tensor:
        """
        Converts an MCTS policy dictionary (move -> prob) to a dense tensor.
        """
        policy_tensor = torch.zeros(action_space_size, device=self.device)
        if not mcts_policy_dict:
            return policy_tensor

        for move, prob in mcts_policy_dict.items():
            try:
                # The board context isn't strictly necessary here if the action space is universal
                idx = move_to_index(move, board=None)
                policy_tensor[idx] = prob
            except (IndexError, Exception) as e:
                print(f"Warning (Trainer): Error converting move {move.uci()} to index: {e}. Skipping.")
        return policy_tensor

    def train_on_batch(self, batch_data: List[Tuple[Tuple, Dict[chess.Move, float], float]], batch_size: int):
        """
        Performs a single training step on a batch of data.
        """
        if not batch_data:
            return 0.0, 0.0

        self.network.train()
        
        # Shuffle data for stochasticity in batches
        indices = torch.randperm(len(batch_data))
        batch_data = [batch_data[i] for i in indices]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for i in range(0, len(batch_data), batch_size):
            self.optimizer.zero_grad()
            
            batch_chunk = batch_data[i:i+batch_size]
            if not batch_chunk:
                continue

            gnn_input_tuples, mcts_policies, game_outcomes = zip(*batch_chunk)

            # NOTE: This loop processes one state at a time within a batch before backprop.
            # This is not fully vectorized batching due to graph data complexity,
            # but it accumulates gradients for a "mini-batch" before updating weights.
            
            batch_policy_loss = 0.0
            batch_value_loss = 0.0

            for j in range(len(gnn_input_tuples)):
                gnn_input_tuple = gnn_input_tuples[j]
                mcts_policy_dict = mcts_policies[j]
                game_outcome = game_outcomes[j]

                processed_gnn_input = tuple(tensor.to(self.device) for tensor in gnn_input_tuple)
                action_space_size = get_action_space_size()
                policy_target = self._convert_mcts_policy_to_tensor(mcts_policy_dict, action_space_size)
                value_target = torch.tensor([game_outcome], dtype=torch.float32, device=self.device)

                pred_policy_logits, pred_value = self.network(*processed_gnn_input)

                # Policy loss: Cross-entropy between predicted logits and MCTS policy distribution
                policy_loss = F.cross_entropy(pred_policy_logits.unsqueeze(0), policy_target.unsqueeze(0))
                
                # Value loss: MSE between predicted value and actual game outcome
                value_loss = self.value_criterion(pred_value, value_target)
                
                # We scale the loss by the number of samples in the chunk before backprop
                total_loss = (policy_loss + value_loss) / len(gnn_input_tuples)
                total_loss.backward()

                batch_policy_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
            
            self.optimizer.step()

            total_policy_loss += batch_policy_loss
            total_value_loss += batch_value_loss
            num_batches += 1

        num_samples = len(batch_data)
        avg_policy_loss = total_policy_loss / num_samples if num_samples > 0 else 0
        avg_value_loss = total_value_loss / num_samples if num_samples > 0 else 0
        
        return avg_policy_loss, avg_value_loss
    
    # --- Checkpointing Methods ---

    def save_checkpoint(self, directory: Path, game_number: int):
        """
        Saves the model, optimizer state, and model configuration to a checkpoint file.
        """
        if not directory.is_dir():
            directory.mkdir(parents=True)

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

    def load_checkpoint(self, directory: Path) -> int:
        """
        Loads the most recent checkpoint from a directory.
        """
        if not directory.is_dir():
            print(f"Checkpoint directory not found: {directory}")
            return 0

        files = [f for f in directory.glob('*.pth.tar')]
        if not files:
            print("No checkpoints found.")
            return 0

        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        try:
            print(f"Loading checkpoint: {latest_file}")
            checkpoint = torch.load(latest_file, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            game_number = checkpoint.get('game_number', 0)

            self.network.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            print(f"Checkpoint loaded successfully from {latest_file}")
            return game_number
        except Exception as e:
            print(f"Error loading checkpoint from {latest_file}: {e}")
            return 0