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
    echo "Stockfish not found. Downloading and setting up Stockfish 15.1..."
    # Create the directory
    mkdir -p $STOCKFISH_DIR
    
    # DEFINITIVE FIX #3: Using a permanent, direct download link from a reliable community mirror.
    wget https://abrok.eu/stockfish/stockfish-15.1-linux-x86-64-avx2.zip -O stockfish.zip
    
    # Unzip the contents, placing the executable directly in our target directory
    # The path inside the zip file is updated to match version 15.1.
    unzip -j stockfish.zip "stockfish-15.1-linux-x86-64-avx2/stockfish_15.1_x64_avx2" -d $STOCKFISH_DIR
    
    # The unzipped file has a versioned name, so we rename it to the generic 'stockfish'
    mv "${STOCKFISH_DIR}/stockfish_15.1_x64_avx2" "$STOCKFISH_EXEC"

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
