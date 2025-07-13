#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Environment Setup ---"

# --- 1. System Dependencies ---
echo "Updating package list and installing system dependencies..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick wget unzip stockfish
echo "System dependencies installed."

# --- 2. Install PyTorch and PyG Ecosystem ---
# This process is now fully environment-aware (CPU vs GPU).

# STEP 2.A: Determine device and set installation URLs.
DEVICE="cpu"
TORCH_URL="" # Default to standard pip repository
PYG_URL="https://data.pyg.org/whl/torch-2.3.1+cpu.html"

# Check if nvidia-smi command exists, indicating a GPU environment
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Setting up for CUDA."
    DEVICE="cu121" # Use CUDA 12.1 as it's a common standard for torch 2.3.1
    TORCH_URL="--index-url https://download.pytorch.org/whl/cu121"
    PYG_URL="https://data.pyg.org/whl/torch-2.3.1+${DEVICE}.html"
else
    echo "No NVIDIA GPU detected. Setting up for CPU."
fi

# STEP 2.B: Install a stable version of PyTorch.
echo "Installing stable PyTorch (v2.3.1) for ${DEVICE}..."
pip install torch==2.3.1 ${TORCH_URL}
echo "PyTorch installed."

# STEP 2.C: Install PyTorch Geometric and its dependencies using the correct URL.
echo "Installing PyTorch Geometric for ${DEVICE}..."
pip install torch-scatter torch-sparse torch-geometric -f ${PYG_URL}
echo "PyTorch Geometric packages installed."

# --- 3. Standard Python Packages ---
echo "Installing remaining Python packages from requirements.txt..."
pip install -r requirements.txt

echo "✅ Environment setup complete."