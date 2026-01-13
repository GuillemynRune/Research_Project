"""Intent classification using pattern matching for robot commands."""
import re
from typing import Dict, List, Tuple

class IntentClassifier:
    """Simple intent classifier for robot commands."""
    
    INTENTS = {
        'vision_query': [
            r'\b(see|look|show|view|camera|what|describe)\b.*\b(you|me|this|that|there|front|around)\b',
            r'\bwhat.*\b(see|seeing)\b',
        ],
        'movement_request': [
            r'\b(move|turn|look|face|point)\b.*\b(left|right|up|down|front)\b',
            r'\b(dance|wave|nod)\b',
        ],
        'emotion_request': [
            r'\b(show|express|feel|be)\b.*\b(happy|sad|curious|excited|surprised)\b',
        ],
        'greeting': [
            r'\b(hello|hi|hey|greetings)\b',
        ],
        'question': [
            r'\b(what|when|where|why|how|who)\b',
            r'\?$',
        ],
        'task_request': [
            r'\b(set|create|start|stop)\b.*\b(timer|reminder|alarm)\b',
        ],
    }
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify user intent from text."""
        text_lower = text.lower()
        
        for intent, patterns in self.INTENTS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return (intent, 0.9)
        
        return ('general_query', 0.5)
    
    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        """Extract entities based on intent."""
        entities = {}
        text_lower = text.lower()
        
        if intent == 'movement_request':
            directions = ['left', 'right', 'up', 'down', 'front']
            for direction in directions:
                if direction in text_lower:
                    entities['direction'] = direction
                    break
        
        elif intent == 'emotion_request':
            emotions = ['happy', 'sad', 'curious', 'excited', 'surprised', 'thinking']
            for emotion in emotions:
                if emotion in text_lower:
                    entities['emotion'] = emotion
                    break
        
        elif intent == 'task_request':
            if 'timer' in text_lower or 'reminder' in text_lower:
                entities['task_type'] = 'timer' if 'timer' in text_lower else 'reminder'
                
                numbers = re.findall(r'\d+', text)
                if numbers:
                    entities['duration'] = numbers[0]
                    
                    if 'second' in text_lower:
                        entities['unit'] = 'seconds'
                    elif 'minute' in text_lower:
                        entities['unit'] = 'minutes'
                    elif 'hour' in text_lower:
                        entities['unit'] = 'hours'
        
        return entities