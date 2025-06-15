#
# File: inspect_checkpoint.py
#
"""
A simple utility to inspect a PyTorch checkpoint file and print the shape of key
layers to determine the model's architecture parameters.
"""
import torch
import argparse
import os

def inspect_checkpoint(checkpoint_path: str):
    """
    Loads a checkpoint and prints the shapes of key layers.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at: {checkpoint_path}")
        return

    print(f"Inspecting checkpoint: {checkpoint_path}\n")

    # Load the checkpoint onto the CPU
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Determine the correct key for the state dictionary
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # --- Key Layers for Parameter Deduction ---
    # We inspect layers whose shapes directly depend on the GNN feature dimensions.

    # 1. PieceGNN's first linear layer weight: [hidden_features, input_features]
    #    The size of the first dimension tells us GNN_HIDDEN_FEATURES.
    piece_conv1_weight_shape = state_dict.get('piece_gnn.conv1.lin.weight', 'Not Found').shape
    print(f"Shape of 'piece_gnn.conv1.lin.weight': {piece_conv1_weight_shape}")
    if piece_conv1_weight_shape != 'Not Found':
        print(f"==> Deduced GNN_HIDDEN_FEATURES: {piece_conv1_weight_shape[0]}")


    # 2. Cross-attention output projection weight: [output_features, output_features]
    #    The size of the dimensions tells us GNN_OUTPUT_FEATURES.
    attention_out_proj_shape = state_dict.get('cross_attention.multi_head_attention.out_proj.weight', 'Not Found').shape
    print(f"\nShape of 'cross_attention.multi_head_attention.out_proj.weight': {attention_out_proj_shape}")
    if attention_out_proj_shape != 'Not Found':
         print(f"==> Deduced GNN_OUTPUT_FEATURES: {attention_out_proj_shape[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inspect a model checkpoint to find its architecture."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth.tar file)."
    )
    args = parser.parse_args()
    inspect_checkpoint(args.model_path)
