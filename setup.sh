#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Environment Setup ---"

# --- 1. System Dependencies ---
echo "Updating package list and installing system dependencies..."
sudo apt-get update
# Use DEBIAN_FRONTEND=noninteractive to prevent interactive prompts during installation
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick wget unzip stockfish

echo "System dependencies installed."

# --- 2. PyTorch and PyG Dependencies ---
# Install PyTorch and its ecosystem libraries separately to ensure correct versions for the environment.
# This avoids the "No matching distribution found" error for pre-compiled wheels.
# The find-links flag points pip to the correct wheel repository for the current PyTorch/CUDA version.
echo "Installing PyTorch, PyG, and related packages..."
pip install torch torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html

echo "PyTorch and PyG packages installed."

# --- 3. Standard Python Packages ---
# The requirements.txt file should NOT contain torch, torch-geometric, torch-scatter,
# or any nvidia-* packages, as they are handled above.
echo "Installing remaining Python packages from requirements.txt..."
pip install -r requirements.txt

echo "✅ Environment setup complete."

