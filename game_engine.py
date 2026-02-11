"""Game engine for D&D simulator."""
import random
from character import Character


class GameEngine:
    """Manages game state and mechanics."""
    
    def __init__(self):
        """Initialize the game engine."""
        self.player = None
        self.game_over = False
        self.available_items = ["potion", "sword", "shield", "helmet", "boots"]
        self.enemies = ["goblin", "orc", "troll", "dragon"]
        
    def start_game(self, player_name):
        """Start a new game with the player."""
        self.player = Character(player_name)
        self.game_over = False
        return f"Welcome, {player_name}! Your adventure begins..."
    
    def attack_enemy(self, enemy_name):
        """Attack an enemy and receive counter-attack."""
        if not self.player.is_alive:
            return "You have been slain! Game Over."
        
        # Player attacks
        player_damage = random.randint(10, 30)
        message = f"You attack the {enemy_name} for {player_damage} damage!\n"
        
        # Enemy counter-attacks
        enemy_damage = random.randint(5, 25)
        self.player.take_damage(enemy_damage)
        message += f"The {enemy_name} strikes back for {enemy_damage} damage!\n"
        
        if not self.player.is_alive:
            message += "\nYou have been slain! Game Over."
            self.game_over = True
        
        return message
    
    def use_item(self, item_name):
        """Use an item from inventory."""
        if not self.player.has_item(item_name):
            return f"You don't have a {item_name}!"
        
        if item_name == "potion":
            heal_amount = 30
            self.player.heal(heal_amount)
            self.player.remove_item(item_name)
            return f"You drink the potion and heal {heal_amount} HP!"
        else:
            return f"You equip the {item_name}."
    
    def pickup_item(self, item_name):
        """Pick up an item."""
        if item_name in self.available_items:
            self.player.add_item(item_name)
            return f"You picked up a {item_name}!"
        else:
            return f"There is no {item_name} here."
    
    def get_player_status(self):
        """Get player status."""
        if self.player:
            return self.player.get_status()
        return "No active character."
