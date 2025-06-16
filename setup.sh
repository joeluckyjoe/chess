#!/bin/bash

# This script sets up the complete environment for the MCTS RL Chess Agent.
# It installs system dependencies, Python packages, and the Stockfish chess engine.

set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Starting Environment Setup ---"

# --- 1. Install System-Level Dependencies ---
echo "Updating package list and installing dependencies (imagemagick, wget, unzip)..."
sudo apt-get update
sudo apt-get install -y imagemagick wget unzip
echo "System dependencies installed."


# --- 2. Install Stockfish using the system package manager ---
echo "Installing Stockfish via apt-get for maximum reliability..."
sudo apt-get install -y stockfish
echo "Stockfish setup complete."


# --- 3. Install Python Packages ---
echo "Installing Python packages from requirements.txt..."

# Get the absolute path of the directory where the script is located.
# This makes the script runnable from any directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REQUIREMENTS_PATH="${SCRIPT_DIR}/requirements.txt"

if [ -f "$REQUIREMENTS_PATH" ]; then
    pip install -r "$REQUIREMENTS_PATH"
else
    echo "Warning: requirements.txt not found at ${REQUIREMENTS_PATH}."
fi
echo "Python package installation complete."


echo "--- ✅ Environment Setup Finished ---"
