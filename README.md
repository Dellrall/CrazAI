# CrazAI - D&D Adventure Simulator

A fun TUI (Terminal User Interface) based D&D simulator with Natural Language Processing written in Python.

## Features

- **Natural Language Processing**: Use natural language commands to play the game
- **Combat System**: Fight enemies like goblins, orcs, trolls, and dragons
- **Health & Damage**: Track your health and take/deal damage
- **Death Mechanics**: Game over when health reaches 0
- **Inventory System**: Pick up and use items like potions, swords, shields, etc.
- **Text-Based Interface**: Beautiful ASCII-based TUI for an immersive experience

## Requirements

- Python 3.7+
- spaCy NLP library
- English language model for spaCy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dellrall/CrazAI.git
cd CrazAI
```

2. Run the setup script:
```bash
bash setup.sh
```

Or manually:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## How to Play

Run the game:
```bash
python game.py
```

### Example Commands

The game understands natural language! Try commands like:

- **Combat**: 
  - "attack the goblin"
  - "fight the dragon"
  - "hit the orc"

- **Inventory**:
  - "pick up the potion"
  - "get the sword"
  - "use potion"
  - "drink potion"

- **Status**:
  - "check status"
  - "show inventory"
  - "what's my health?"

- **Other**:
  - "help" - Show available commands
  - "quit" - Exit the game

## Project Information

- **Duration**: 10 week project
- **Team Size**: 3 person project
- **Purpose**: Introduction to AI assignment
- **Technologies**: Python, spaCy NLP library
- **Design**: Short and fun gameplay with core features

## License

See LICENSE file for details.
