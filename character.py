"""Character class for D&D simulator."""


class Character:
    """Represents a player character in the game."""
    
    def __init__(self, name, max_health=100):
        """Initialize a character with name and health."""
        self.name = name
        self.max_health = max_health
        self.health = max_health
        self.inventory = []
        self.is_alive = True
        
    def take_damage(self, damage):
        """Apply damage to character and check if dead."""
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
        return self.health
    
    def heal(self, amount):
        """Heal the character."""
        self.health = min(self.health + amount, self.max_health)
        return self.health
    
    def add_item(self, item):
        """Add an item to inventory."""
        self.inventory.append(item)
        
    def remove_item(self, item):
        """Remove an item from inventory."""
        if item in self.inventory:
            self.inventory.remove(item)
            return True
        return False
    
    def has_item(self, item):
        """Check if character has an item."""
        return item in self.inventory
    
    def get_status(self):
        """Return character status as string."""
        status = f"\n{self.name}'s Status:\n"
        status += f"Health: {self.health}/{self.max_health}\n"
        status += f"Status: {'Alive' if self.is_alive else 'Dead'}\n"
        status += f"Inventory ({len(self.inventory)} items): "
        if self.inventory:
            status += ", ".join(self.inventory)
        else:
            status += "Empty"
        return status
