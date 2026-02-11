#!/usr/bin/env python3
"""Main game file for D&D TUI simulator."""
import sys
from game_engine import GameEngine
from nlp_parser import NLPParser


class DnDGame:
    """Main game class handling the TUI and game loop."""
    
    def __init__(self):
        """Initialize the game."""
        self.engine = GameEngine()
        self.parser = NLPParser()
        self.running = False
        
    def display_welcome(self):
        """Display welcome message."""
        print("\n" + "="*60)
        print("  WELCOME TO THE D&D ADVENTURE SIMULATOR")
        print("="*60)
        print("\nA text-based adventure with natural language processing!")
        print("\nYou can use natural language commands like:")
        print("  - 'attack the goblin'")
        print("  - 'pick up the potion'")
        print("  - 'use potion'")
        print("  - 'check status'")
        print("  - 'help' for more commands")
        print("\nType 'quit' to exit the game.")
        print("="*60 + "\n")
    
    def display_help(self):
        """Display help message."""
        help_text = """
╔═══════════════════════════════════════════════════════════╗
║                     COMMAND HELP                          ║
╠═══════════════════════════════════════════════════════════╣
║ COMBAT:                                                   ║
║   attack [enemy] - Attack an enemy (goblin, orc, etc.)   ║
║   fight [enemy]  - Same as attack                        ║
║                                                           ║
║ INVENTORY:                                                ║
║   pick up [item] - Pick up an item                       ║
║   get [item]     - Same as pick up                       ║
║   use [item]     - Use an item from inventory            ║
║                                                           ║
║ STATUS:                                                   ║
║   status         - Check your health and inventory       ║
║   check stats    - Same as status                        ║
║                                                           ║
║ OTHER:                                                    ║
║   help           - Show this help message                ║
║   quit           - Exit the game                         ║
╚═══════════════════════════════════════════════════════════╝
        """
        print(help_text)
    
    def process_command(self, command):
        """Process a parsed command."""
        intent = command["intent"]
        target = command["target"]
        
        if intent == "attack":
            if target:
                return self.engine.attack_enemy(target)
            else:
                return "Attack what? (e.g., 'attack goblin')"
        
        elif intent == "pickup":
            if target:
                return self.engine.pickup_item(target)
            else:
                return "Pick up what? (e.g., 'pick up potion')"
        
        elif intent == "use":
            if target:
                return self.engine.use_item(target)
            else:
                return "Use what? (e.g., 'use potion')"
        
        elif intent == "status":
            return self.engine.get_player_status()
        
        elif intent == "help":
            self.display_help()
            return ""
        
        elif intent == "quit":
            return "QUIT"
        
        else:
            return "I don't understand that command. Type 'help' for available commands."
    
    def run(self):
        """Main game loop."""
        self.display_welcome()
        
        # Get player name
        player_name = input("What is your name, adventurer? ").strip()
        if not player_name:
            player_name = "Hero"
        
        # Start game
        print("\n" + self.engine.start_game(player_name))
        print(self.engine.get_player_status())
        
        self.running = True
        
        # Main game loop
        while self.running:
            try:
                # Get user input
                print("\n" + "-"*60)
                user_input = input("What do you do? > ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                command = self.parser.parse_command(user_input)
                
                # Process command
                result = self.process_command(command)
                
                if result == "QUIT":
                    print("\nThanks for playing! Goodbye!")
                    self.running = False
                    break
                
                # Display result
                if result:
                    print("\n" + result)
                
                # Check game over
                if self.engine.game_over:
                    print("\n" + "="*60)
                    print("           GAME OVER")
                    print("="*60)
                    self.running = False
                
            except KeyboardInterrupt:
                print("\n\nGame interrupted. Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Type 'help' for available commands.")


def main():
    """Entry point for the game."""
    game = DnDGame()
    game.run()


if __name__ == "__main__":
    main()
