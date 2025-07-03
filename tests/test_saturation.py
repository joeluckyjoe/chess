# FILE: tests/test_saturation.py
import pytest
import torch
import chess
from typing import Dict, Any

from gnn_agent.rl_loop.trainer import Trainer
from gnn_agent.neural_network.chess_network import ChessNetwork
from gnn_agent.gamestate_converters.gnn_data_converter import convert_to_gnn_input
from config import config_params

def calculate_gradient_saturation(network: torch.nn.Module, epsilon: float = 1e-8) -> float:
    """
    Calculates the percentage of gradients in the network that are close to zero.
    A high percentage suggests a vanishing gradient problem (neuron saturation).

    Args:
        network: The PyTorch network module to inspect.
        epsilon: The threshold below which a gradient is considered "dead".

    Returns:
        The percentage of saturated (dead) gradients.
    """
    total_params = 0
    saturated_params = 0
    for param in network.parameters():
        if param.grad is not None:
            grad_abs = torch.abs(param.grad.detach())
            total_params += grad_abs.numel()
            saturated_params += (grad_abs < epsilon).sum().item()

    if total_params == 0:
        return 0.0  # No gradients were found

    return (saturated_params / total_params) * 100.0


def test_network_health_and_convergence():
    """
    A two-part test for network health:
    1. Health Check: Validates that on the first backward pass (when loss is high),
       gradient saturation is low, proving healthy gradient flow.
    2. Convergence Check: Validates that the network can successfully overfit
       on a single data point, proving its capacity to learn.
    """
    # 1. SETUP
    board = chess.Board()
    device = torch.device("cpu")
    training_example = [(board.fen(), {chess.Move.from_uci("e2e4"): 1.0}, 0.5)]
    
    # 2. INITIALIZATION
    test_config = config_params.copy()
    test_config['LEARNING_RATE'] = 0.01
    test_config['WEIGHT_DECAY'] = 0
    test_config['LR_SCHEDULER_STEP_SIZE'] = 1_000_000 

    trainer = Trainer(model_config=test_config, device=device)
    network, _ = trainer.load_or_initialize_network(directory=None)
    network.train()

    # 3. PART 1: INITIAL HEALTH CHECK
    print("\n--- Running Part 1: Initial Gradient Health Check ---")
    
    # Perform a single training step
    initial_policy_loss, _ = trainer.train_on_batch(
        game_examples=training_example, puzzle_examples=[], batch_size=1
    )
    
    # Check saturation immediately after the first step
    initial_saturation = calculate_gradient_saturation(network)
    
    print(f"Initial Policy Loss: {initial_policy_loss:.6f}")
    print(f"Initial Gradient Saturation: {initial_saturation:.2f}%")
    
    assert initial_saturation < 10.0, f"High initial gradient saturation detected ({initial_saturation:.2f}%). Network has a health problem."
    print("SUCCESS: Initial gradient flow is healthy.")

    # 4. PART 2: CONVERGENCE CHECK
    print("\n--- Running Part 2: Overfitting Convergence Check ---")
    num_overfit_iterations = 100 # We already did one, so 99 more.
    final_policy_loss = initial_policy_loss
    final_value_loss = -1.0
    
    for i in range(num_overfit_iterations - 1):
        policy_loss, value_loss = trainer.train_on_batch(
            game_examples=training_example, puzzle_examples=[], batch_size=1
        )
        final_policy_loss = policy_loss
        final_value_loss = value_loss
        if (i + 10) % 10 == 0:
             print(f"Iteration {i+10:3d}/{num_overfit_iterations} -> Policy Loss: {policy_loss:.6f}, Value Loss: {value_loss:.6f}")

    print("--- Overfitting loop complete ---")
    print(f"Final Policy Loss: {final_policy_loss:.6f}")
    print(f"Final Value Loss: {final_value_loss:.6f}")
    
    assert final_policy_loss < 0.1, f"Policy loss did not converge. Final loss: {final_policy_loss}"
    assert final_value_loss < 0.1, f"Value loss did not converge. Final loss: {final_value_loss}"
    print("SUCCESS: Network can successfully converge on a target.")