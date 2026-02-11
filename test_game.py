#!/usr/bin/env python3
"""Simple tests for D&D simulator components."""

from character import Character
from game_engine import GameEngine
from nlp_parser import NLPParser


def test_character():
    """Test character creation and basic functions."""
    print("Testing Character class...")
    char = Character("Test Hero", max_health=100)
    assert char.name == "Test Hero"
    assert char.health == 100
    assert char.is_alive == True
    
    # Test damage
    char.take_damage(30)
    assert char.health == 70
    
    # Test healing
    char.heal(20)
    assert char.health == 90
    
    # Test death
    char.take_damage(100)
    assert char.health == 0
    assert char.is_alive == False
    
    print("✓ Character class tests passed")


def test_inventory():
    """Test inventory system."""
    print("Testing inventory system...")
    char = Character("Test")
    
    # Add items
    char.add_item("sword")
    char.add_item("potion")
    assert len(char.inventory) == 2
    assert char.has_item("sword")
    
    # Remove items
    char.remove_item("sword")
    assert len(char.inventory) == 1
    assert not char.has_item("sword")
    
    print("✓ Inventory tests passed")


def test_nlp_parser():
    """Test NLP parser."""
    print("Testing NLP parser...")
    parser = NLPParser()
    
    # Test attack commands
    cmd = parser.parse_command("attack the goblin")
    assert cmd["intent"] == "attack"
    assert cmd["target"] == "goblin"
    
    # Test pickup commands
    cmd = parser.parse_command("pick up the potion")
    assert cmd["intent"] == "pickup"
    assert cmd["target"] == "potion"
    
    # Test use commands
    cmd = parser.parse_command("use potion")
    assert cmd["intent"] == "use"
    assert cmd["target"] == "potion"
    
    # Test status command
    cmd = parser.parse_command("check status")
    assert cmd["intent"] == "status"
    
    print("✓ NLP parser tests passed")


def test_game_engine():
    """Test game engine."""
    print("Testing game engine...")
    engine = GameEngine()
    
    # Start game
    result = engine.start_game("Hero")
    assert engine.player is not None
    assert engine.player.name == "Hero"
    
    # Pickup item
    result = engine.pickup_item("potion")
    assert "picked up" in result.lower()
    assert engine.player.has_item("potion")
    
    # Use item
    engine.player.health = 50
    result = engine.use_item("potion")
    assert "heal" in result.lower()
    assert engine.player.health == 80  # 50 + 30
    
    print("✓ Game engine tests passed")


if __name__ == "__main__":
    print("="*60)
    print("Running D&D Simulator Tests")
    print("="*60 + "\n")
    
    test_character()
    test_inventory()
    test_nlp_parser()
    test_game_engine()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
