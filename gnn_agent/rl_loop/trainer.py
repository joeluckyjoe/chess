# gnn_agent/rl_loop/trainer.py (Updated)

import torch
import os
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
from typing import Dict, List, Tuple, Any, Optional # Added Optional
import chess

from ..neural_network.chess_network import ChessNetwork
from ..gamestate_converters.action_space_converter import move_to_index, get_action_space_size

class Trainer:
    """
    Manages the training process for the ChessNetwork.
    """
    # --- MODIFIED: Added model_config to __init__ ---
    def __init__(self, network: ChessNetwork, model_config: Dict[str, Any], learning_rate: float = 0.001, device: torch.device = torch.device("cpu")):
        """
        Initializes the Trainer.

        Args:
            network (ChessNetwork): The neural network model to be trained.
            model_config (Dict[str, Any]): Dictionary containing the model's architectural parameters.
            learning_rate (float): The learning rate for the optimizer.
            device (torch.device): The device to run training on (cpu or cuda).
        """
        self.network = network
        self.model_config = model_config # --- NEW: Store the model configuration ---
        self.device = device
        self.network.to(self.device)
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
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
                idx = move_to_index(move, board=None) # Assuming context not needed
                policy_tensor[idx] = prob
            except (IndexError, Exception) as e:
                print(f"Warning (Trainer): Error converting move {move.uci()} to index: {e}. Skipping.")
        return policy_tensor

    def train_on_batch(self, batch_data: List[Tuple[Tuple, Dict[chess.Move, float], float]]):
        """
        Performs a single training step on a batch of data.
        """
        if not batch_data:
            return 0.0, 0.0

        gnn_input_tuples_list, mcts_policy_dicts_list, game_outcomes_list = zip(*batch_data)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        self.network.train()

        # NOTE: This loop processes one state at a time. For larger-scale training,
        # you might investigate batching graph data using torch_geometric.data.Batch
        # which would require changes to the forward pass. For now, this is correct.
        for i in range(len(gnn_input_tuples_list)):
            gnn_input_tuple = gnn_input_tuples_list[i]
            mcts_policy_dict = mcts_policy_dicts_list[i]
            game_outcome = game_outcomes_list[i]

            processed_gnn_input = tuple(tensor.to(self.device) for tensor in gnn_input_tuple)
            action_space_size = get_action_space_size()
            policy_target_tensor = self._convert_mcts_policy_to_tensor(mcts_policy_dict, action_space_size)
            value_target_tensor = torch.tensor([game_outcome], dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()

            pred_policy_logits, pred_value = self.network(*processed_gnn_input)

            # Policy loss: Cross-entropy between predicted logits and MCTS policy distribution
            policy_loss = F.cross_entropy(pred_policy_logits.unsqueeze(0), policy_target_tensor.unsqueeze(0))
            
            # Value loss: MSE between predicted value and actual game outcome
            value_loss = self.value_criterion(pred_value, value_target_tensor)

            current_total_loss = policy_loss + value_loss
            current_total_loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        num_samples = len(gnn_input_tuples_list)
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

        # --- MODIFIED: Added 'config_params' to the state dictionary ---
        state = {
            'game_number': game_number,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config_params': self.model_config, # Save the network architecture config
        }

        torch.save(state, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, directory: Path) -> int:
        """
        Loads the most recent checkpoint from a directory.
        Note: This method assumes the network object already has the correct architecture.
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

            # Note: We don't need to use 'config_params' here because we are loading
            # into a pre-existing `self.network` instance. The evaluation script, however,
            # will need 'config_params' to build the network from scratch.

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