#!/bin/bash
# Setup script for D&D simulator

echo "Setting up D&D Simulator..."

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

echo "Setup complete! Run 'python game.py' to start the game."
