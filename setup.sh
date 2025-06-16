#!/bin/bash

# This script sets up the complete environment for the MCTS RL Chess Agent.
# It installs system dependencies, Python packages, and the correct Stockfish version.

set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Starting Environment Setup ---"

# --- 1. Install System-Level Dependencies ---
echo "Updating package list and installing dependencies (imagemagick, wget, unzip)..."
sudo apt-get update
sudo apt-get install -y imagemagick wget unzip
echo "System dependencies installed."


# --- 2. Install Stockfish using the system package manager ---
# REVERTED: Using the simple, robust 'apt-get' method as it is more reliable
# than direct downloads which have been failing.
echo "Installing Stockfish via apt-get for maximum reliability..."
sudo apt-get install -y stockfish
echo "Stockfish setup complete."


# --- 3. Install Python Packages ---
echo "Installing Python packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Please ensure all required Python packages are installed."
fi
echo "Python package installation complete."


echo "--- ✅ Environment Setup Finished ---"
