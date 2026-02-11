"""NLP parser for understanding player commands."""
import spacy


class NLPParser:
    """Parses natural language commands using spaCy."""
    
    def __init__(self):
        """Initialize the NLP parser."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, will use simple parsing
            self.nlp = None
    
    def parse_command(self, text):
        """Parse a command and extract intent and entities."""
        text = text.lower().strip()
        
        # Simple keyword-based parsing
        command = {
            "intent": None,
            "target": None,
            "raw": text
        }
        
        # Attack commands
        if any(word in text for word in ["attack", "fight", "hit", "strike", "kill"]):
            command["intent"] = "attack"
            # Extract target after attack verb
            for word in ["goblin", "orc", "troll", "dragon", "enemy"]:
                if word in text:
                    command["target"] = word
                    break
        
        # Pick up commands
        elif any(word in text for word in ["pick", "get", "take", "grab", "pickup"]):
            command["intent"] = "pickup"
            # Extract item
            for word in ["potion", "sword", "shield", "helmet", "boots"]:
                if word in text:
                    command["target"] = word
                    break
        
        # Use item commands
        elif any(word in text for word in ["use", "drink", "equip", "wear"]):
            command["intent"] = "use"
            for word in ["potion", "sword", "shield", "helmet", "boots"]:
                if word in text:
                    command["target"] = word
                    break
        
        # Status commands
        elif any(word in text for word in ["status", "stats", "health", "inventory", "check"]):
            command["intent"] = "status"
        
        # Help commands
        elif any(word in text for word in ["help", "commands", "?"]):
            command["intent"] = "help"
        
        # Quit commands
        elif any(word in text for word in ["quit", "exit", "bye", "goodbye"]):
            command["intent"] = "quit"
        
        # Use spaCy for more advanced parsing if available
        if self.nlp and command["intent"] is None:
            doc = self.nlp(text)
            # Extract verbs and nouns
            verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            
            # Map common verb lemmas to intents
            verb_to_intent = {
                "attack": "attack", "fight": "attack", "hit": "attack", 
                "strike": "attack", "kill": "attack",
                "pick": "pickup", "get": "pickup", "take": "pickup", 
                "grab": "pickup",
                "use": "use", "drink": "use", "equip": "use", "wear": "use"
            }
            
            if verbs and verbs[0] in verb_to_intent:
                command["intent"] = verb_to_intent[verbs[0]]
            if nouns:
                command["target"] = nouns[0]
        
        return command
