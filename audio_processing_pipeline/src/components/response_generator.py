import asyncio
import json
import os
import random
import traceback
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.api.models import ComponentAPIRequest, ComponentAPIResponse
from src.core.component import Component, ComponentConfig
from src.utils.logger import logger


class ResponseGeneratorConfig(ComponentConfig):
    """Configuration for response generator component"""
    model_name: str = "facebook/opt-125m"  # Lightweight model for testing
    device: str = "cpu"  # 'cpu' or 'cuda' for GPU
    max_length: int = 100
    temperature: float = 0.7
    use_cached_responses: bool = False  # Use cached responses for common inputs
    response_cache_file: str = "cached_responses.json"


class ResponseGenerator(Component):
    """Response generator component using a language model"""

    def __init__(self, config: ResponseGeneratorConfig):
        super().__init__(config)
        self.model_name = config.model_name
        self.device = config.device
        self.max_length = config.max_length
        self.temperature = config.temperature
        self.use_cached_responses = config.use_cached_responses
        self.response_cache_file = config.response_cache_file

        self.model = None
        self.tokenizer = None
        self.response_cache = {}
        self.context = []

    async def initialize(self) -> None:
        """Initialize the language model"""
        logger.info(f"Initializing {self.name} with model {self.model_name}")

        # Load cached responses if enabled
        if self.use_cached_responses:
            await self._load_response_cache()

        try:
            # Initialize the model in a separate thread
            if not self.use_cached_responses:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._load_model
                )
                logger.info(f"Response generator model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load language model: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall back to cached responses
            logger.warning("Falling back to cached responses only")
            self.use_cached_responses = True

        await super().initialize()

    #for lightweight version
    
    async def process(self, data: Dict[str, any]) -> Dict[str, any]:
        """Process the input data and generate a response"""
        # Get the transcription from the input data
        transcription = data.get("transcription", "")

        # Generate a simple response based on the transcription
        response = self._generate_simple_response(transcription)

        # Add the response to the data
        data["response"] = response

        return data

    def _generate_simple_response(self, text: str) -> str:
        """Generate a simple response based on the input text"""
        # Simple rule-based responses
        text = text.lower()

        if not text:
            return "I didn't catch that. Could you please repeat?"

        if "hello" in text or "hi" in text:
            return "Hello there! How can I help you today?"

        if "how are you" in text:
            return "I'm doing well, thank you for asking. How about you?"

        if "weather" in text:
            return "I don't have access to real-time weather data, but I hope it's nice where you are!"

        if "name" in text:
            return "I'm your audio processing assistant. Nice to meet you!"

        if "thank" in text:
            return "You're welcome! Is there anything else I can help with?"

        if "bye" in text or "goodbye" in text:
            return "Goodbye! Have a great day!"

        # Default response
        return f"I heard: {text}. How can I assist you with that?"
    #for lightweight version

    def _load_model(self):
        """Load the language model (runs in a separate thread)"""
        # Check if GPU is available
        device = torch.device(self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu")
        logger.info(f"Using device: {device}")

        # This is where we'd use CUDA for the GPU-intensive model loading
        # CUDA USAGE SECTION START
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(device)  # Move model to GPU if available
        # CUDA USAGE SECTION END

        logger.info(f"Model loaded on {device}")