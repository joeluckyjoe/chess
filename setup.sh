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

# --- 2. Install Stable PyTorch (CPU or GPU) ---
# This section is now a two-step process to ensure correctness.

# STEP 2.A: Install a stable version of PyTorch.
# We will install a known-good, stable version (2.3.1) to ensure compatibility.
echo "Installing stable PyTorch (v2.3.1)..."
pip install torch==2.3.1
echo "PyTorch installed."

# STEP 2.B: Install PyTorch Geometric and its dependencies.
# This command now runs AFTER torch is installed, so the version check will succeed.
# It installs torch-scatter and torch-sparse, which are required by torch-geometric.
echo "Installing PyTorch Geometric and dependencies..."
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.3.1+cpu.html
echo "PyTorch Geometric packages installed."


# --- 3. Standard Python Packages ---
# The requirements.txt file should NOT contain torch, torch-geometric, torch-scatter,
# or any nvidia-* packages, as they are handled above.
echo "Installing remaining Python packages from requirements.txt..."
pip install -r requirements.txt

echo "✅ Environment setup complete."