import asyncio
import os
import traceback
from typing import Dict, List, Optional, Union

import spacy

from src.api.models import ComponentAPIRequest, ComponentAPIResponse
from src.core.component import Component, ComponentConfig
from src.utils.logger import logger


class TextProcessingConfig(ComponentConfig):
    """Configuration for text processing component"""
    model_name: str = "en_core_web_sm"
    agentic: bool = False  # Use more complex context tracking and reasoning
    use_mock: bool = False
    mock_entities: Dict[str, List[str]] = {
        "PERSON": ["John", "Mary", "Steve Jobs"],
        "ORG": ["Apple", "Google", "Microsoft"],
        "GPE": ["New York", "San Francisco", "London"],
    }


class TextProcessing(Component):
    """Text processing component using spaCy"""

    def __init__(self, config: TextProcessingConfig):
        super().__init__(config)
        self.model_name = config.model_name
        self.agentic = config.agentic
        self.use_mock = config.use_mock
        self.mock_entities = config.mock_entities
        self.nlp = None
        self.context = {}  # Store conversation context

    async def initialize(self) -> None:
        """Initialize the NLP model"""
        logger.info(f"Initializing {self.name} with model {self.model_name}")

        if not self.use_mock:
            try:
                # Initialize the spaCy model in a separate thread
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._load_model
                )
                logger.info(f"NLP model {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NLP model: {str(e)}")
                logger.error(traceback.format_exc())
                # Fall back to mock mode
                logger.warning("Falling back to mock mode for NLP")
                self.use_mock = True

        await super().initialize()

    def _load_model(self):
        """Load the spaCy model (runs in a separate thread)"""
        try:
            # Check if the model is already downloaded
            if not spacy.util.is_package(self.model_name):
                logger.info(f"Downloading spaCy model: {self.model_name}")
                os.system(f"python -m spacy download {self.model_name}")

            # Load the model
            self.nlp = spacy.load(self.model_name)
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            raise

    async def process(self, text: str) -> Dict[str, Union[str, Dict]]:
        """Process text with NLP

        Args:
            text: Input text

        Returns:
            Dictionary with processed information
        """
        if not self.initialized:
            raise RuntimeError("Component not initialized")

        if not text:
            return None

        # Use mock mode for testing
        if self.use_mock:
            # Create a mock NLP result
            entities = []
            for entity_type, values in self.mock_entities.items():
                for value in values:
                    if value.lower() in text.lower():
                        entities.append({
                            "text": value,
                            "label": entity_type,
                            "start": text.lower().find(value.lower()),
                            "end": text.lower().find(value.lower()) + len(value)
                        })

            # Simulate processing delay
            await asyncio.sleep(0.1)

            return {
                "original_text": text,
                "entities": entities,
                "sentiment": "positive" if "good" in text.lower() else "negative" if "bad" in text.lower() else "neutral"
            }

        try:
            # Process text with spaCy (in a separate thread)
            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(
                None,
                lambda: self.nlp(text)
            )

            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

            # Extract other information
            result = {
                "original_text": text,
                "entities": entities,
                "sentiment": self._analyze_sentiment(doc),
            }

            if self.agentic:
                # For agentic mode, update and use context
                self._update_context(text, entities)
                result["context"] = self.context
                result["intent"] = self._determine_intent(doc)

            return result
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}")
            logger.error(traceback.format_exc())
            return {"original_text": text, "error": str(e)}