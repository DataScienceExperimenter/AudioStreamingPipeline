import gc
from typing import Dict, Any, List

# Try to import logger, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("text_processing")

class TextProcessing:
    def __init__(self, config=None, performance_tracker=None):
        self.name = "TextProcessing"
        self.performance_tracker = performance_tracker
        self.model_name = config.get("model_name", "en_core_web_sm") if config else "en_core_web_sm"
        self.use_mock = config.get("use_mock", False) if config else False
        self.nlp = None

        logger.info(f"Initializing {self.name}")

    async def initialize(self):
        try:
            if not self.use_mock:
                # Try to import spacy
                try:
                    import spacy

                    # Load spacy model
                    self.nlp = spacy.load(self.model_name)

                    logger.info(f"Loaded spaCy model: {self.model_name}")
                except ImportError:
                    logger.warning("spaCy not installed. Using mock text processing.")
                    self.use_mock = True
                except OSError:
                    logger.warning(f"spaCy model {self.model_name} not found. Using mock text processing.")
                    self.use_mock = True

            logger.info(f"{self.name} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing {self.name}: {str(e)}")
            self.use_mock = True
            return False

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis"""
        # This is a very basic sentiment analysis
        positive_words = ["good", "great", "excellent", "happy", "positive", "nice", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "sad", "negative", "hate", "dislike"]

        text_lower = text.lower()

        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Calculate sentiment score
        if positive_count > negative_count:
            return {"score": 0.5 + (positive_count - negative_count) * 0.1, "label": "positive"}
        elif negative_count > positive_count:
            return {"score": 0.5 - (negative_count - positive_count) * 0.1, "label": "negative"}
        else:
            return {"score": 0.5, "label": "neutral"}

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "as", "of", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "this", "that", "these", "those"}

        # Tokenize text
        words = text.lower().split()

        # Remove common words and punctuation
        keywords = [word.strip(".,!?;:()[]{}\"'") for word in words if word.lower() not in common_words and len(word) > 2]

        # Return unique keywords
        return list(set(keywords))

    def _determine_intent(self, text: str) -> str:
        """Determine the intent of the text"""
        text_lower = text.lower()

        # Check for question
        if "?" in text or any(word in text_lower for word in ["what", "who", "where", "when", "why", "how"]):
            return "question"

        # Check for greeting
        if any(greeting in text_lower for greeting in ["hello", "hi", "hey", "greetings"]):
            return "greeting"

        # Check for farewell
        if any(farewell in text_lower for farewell in ["goodbye", "bye", "see you", "farewell"]):
            return "farewell"

        # Check for help request
        if any(help_word in text_lower for help_word in ["help", "assist", "support"]):
            return "help_request"

        # Check for command
        if text_lower.startswith(("please", "can you", "could you", "would you")):
            return "command"

        # Default to general query
        return "general_query"

    async def process(self, text: str) -> Dict[str, Any]:
        """Process text input"""
        if self.performance_tracker:
            self.performance_tracker.start_component("TextProcessing")

        try:
            # Check if text is empty
            if not text:
                result = {
                    "processed_text": "",
                    "entities": [],
                    "intent": "none",
                    "sentiment": {"score": 0.5, "label": "neutral"},
                    "keywords": []
                }
            else:
                if self.use_mock or not self.nlp:
                    # Process text without spaCy
                    intent = self._determine_intent(text)
                    sentiment = self._analyze_sentiment(text)
                    keywords = self._extract_keywords(text)

                    result = {
                        "processed_text": text,
                        "entities": [],
                        "intent": intent,
                        "sentiment": sentiment,
                        "keywords": keywords
                    }
                else:
                    # Process text with spaCy
                    doc = self.nlp(text)

                    # Extract entities
                    entities = [
                        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                        for ent in doc.ents
                    ]

                    # Determine intent
                    intent = self._determine_intent(text)

                    # Analyze sentiment
                    sentiment = self._analyze_sentiment(text)

                    # Extract keywords
                    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]

                    result = {
                        "processed_text": text,
                        "entities": entities,
                        "intent": intent,
                        "sentiment": sentiment,
                        "keywords": keywords
                    }

            if self.performance_tracker:
                elapsed = self.performance_tracker.end_component("TextProcessing")
                logger.info(f"TextProcessing time: {elapsed:.3f}s")

            # Log the processed text analysis
            print("\n" + "=" * 50)
            print("🧠 TEXT ANALYSIS")
            print("=" * 50)
            print(f"Intent: {result['intent']}")
            print(f"Sentiment: {result['sentiment']['label']} (score: {result['sentiment']['score']:.2f})")
            if result['keywords']:
                print(f"Keywords: {', '.join(result['keywords'][:5])}")
            if result['entities']:
                entities_str = ", ".join([f"{e['text']} ({e['label']})" for e in result['entities'][:3]])
                print(f"Entities: {entities_str}")
            print("=" * 50 + "\n")

            logger.info(f"🧠 TEXT ANALYSIS: Intent: {result['intent']}, Sentiment: {result['sentiment']['label']}")
            if result['keywords']:
                logger.info(f"🔑 KEYWORDS: {', '.join(result['keywords'][:5])}")
            if result['entities']:
                entities_str = ", ".join([f"{e['text']} ({e['label']})" for e in result['entities'][:3]])
                logger.info(f"🏷️ ENTITIES: {entities_str}")

            return result

        except Exception as e:
            logger.error(f"Error in TextProcessing: {str(e)}")

            if self.performance_tracker:
                self.performance_tracker.end_component("TextProcessing")

            return {"processed_text": text, "error": str(e)}

    async def shutdown(self):
        # Clear model from memory
        self.nlp = None

        # Force garbage collection
        gc.collect()

        logger.info(f"{self.name} shutdown successfully")
        return True
