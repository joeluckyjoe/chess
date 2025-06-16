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
    echo "Stockfish not found. Downloading and setting up Stockfish 16.1.1..."
    # Create the directory
    mkdir -p $STOCKFISH_DIR
    
    # DEFINITIVE FIX: Using the stable, direct download link from the official GitHub release.
    wget https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1.1/stockfish-16.1.1-linux-x64-avx2.zip -O stockfish.zip
    
    # Unzip the contents, placing the executable directly in our target directory
    # The path inside the zip file has also been updated to match the new version.
    unzip -j stockfish.zip "stockfish-16.1.1-linux-x64-avx2/stockfish" -d $STOCKFISH_DIR
    
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
