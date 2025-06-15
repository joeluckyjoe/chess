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


# --- 2. Install Specific Stockfish Version ---
STOCKFISH_DIR="stockfish"
STOCKFISH_EXEC="${STOCKFISH_DIR}/stockfish"

if [ -f "$STOCKFISH_EXEC" ]; then
    echo "Stockfish executable already found. Skipping download."
else
    echo "Stockfish not found. Downloading and setting up Stockfish 16..."
    # Create the directory
    mkdir -p $STOCKFISH_DIR
    
    # Download the official Stockfish 16 for AVX2 CPUs (common on modern systems/Colab)
    wget https://stockfishchess.org/files/stockfish_16_linux_x64_avx2.zip -O stockfish.zip
    
    # Unzip the contents
    unzip -j stockfish.zip "stockfish_16_linux_x64_avx2/stockfish" -d $STOCKFISH_DIR
    
    # Clean up the downloaded zip file
    rm stockfish.zip
    
    # Make the binary executable
    chmod +x $STOCKFISH_EXEC
    
    echo "Stockfish setup complete. Executable at: ${STOCKFISH_EXEC}"
fi


# --- 3. Install Python Packages ---
echo "Installing Python packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Please ensure all required Python packages are installed."
fi
echo "Python package installation complete."


echo "--- ✅ Environment Setup Finished ---"