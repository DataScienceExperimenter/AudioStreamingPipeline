import asyncio
import traceback
from typing import List, Optional

import numpy as np
import torch
from transformers import pipeline

from src.api.models import ComponentAPIRequest, ComponentAPIResponse
from src.core.component import Component, ComponentConfig
from src.utils.logger import logger


class SpeechToTextConfig(ComponentConfig):
    """Configuration for speech-to-text component"""
    model_name: str = "facebook/wav2vec2-base-960h"
    language: str = "en-US"
    device: str = "cpu"  # 'cpu' or 'cuda' for GPU if available
    chunk_size: int = 16000  # Process in chunks of 1 second (16000 samples)
    use_mock: bool = False
    mock_texts: List[str] = ["Hello world", "This is a test", "Audio processing pipeline"]


class SpeechToText(Component):
    """Speech-to-text component using Transformers"""

    def __init__(self, config: SpeechToTextConfig):
        super().__init__(config)
        self.model_name = config.model_name
        self.language = config.language
        self.device = config.device
        self.chunk_size = config.chunk_size
        self.use_mock = config.use_mock
        self.mock_texts = config.mock_texts
        self.mock_index = 0
        self.model = None
        self.processor = None
        self.asr_pipeline = None
        self.buffer = b''  # Buffer for accumulating audio chunks

    async def initialize(self) -> None:
        """Initialize the STT model"""
        logger.info(f"Initializing {self.name} with model {self.model_name}")

        if not self.use_mock:
            try:
                # Initialize the ASR pipeline with Transformers
                # Using a separate thread to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._load_model
                )
                logger.info(f"STT model {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load STT model: {str(e)}")
                logger.error(traceback.format_exc())
                # Fall back to mock mode
                logger.warning("Falling back to mock mode for STT")
                self.use_mock = True

        await super().initialize()

    def _load_model(self):
        """Load the ASR model (runs in a separate thread)"""
        # Check if GPU is available
        device = torch.device(self.device if torch.cuda.is_available() and self.device == "cuda" else "cpu")

        # Load models directly
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=device
        )

        logger.info(f"Device set to use {device}")

    async def process(self, audio_data: bytes) -> str:
        """Convert speech to text

        Args:
            audio_data: Raw audio bytes

        Returns:
            Transcribed text
        """
        if not self.initialized:
            raise RuntimeError("Component not initialized")

        if not audio_data:
            return None

        # Use mock mode for testing
        if self.use_mock:
            # Cycle through mock texts
            text = self.mock_texts[self.mock_index]
            self.mock_index = (self.mock_index + 1) % len(self.mock_texts)
            # Simulate processing delay
            await asyncio.sleep(0.2)
            return text

        try:
            # Add current chunk to buffer
            self.buffer += audio_data

            # Only process if we have enough data
            if len(self.buffer) >= self.chunk_size * 2:  # 16-bit audio
                # Convert bytes to numpy array
                audio_array = np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32) / 32767.0

                # Clear buffer
                self.buffer = b''

                # Process audio with ASR pipeline (in a separate thread)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.asr_pipeline(audio_array)
                )

                # Extract transcribed text
                text = result["text"] if "text" in result else ""
                return text.strip()
            else:
                return None
        except Exception as e:
            logger.error(f"Error in STT processing: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def handle_api_request(self, request: ComponentAPIRequest) -> ComponentAPIResponse:
        """Handle API requests specific to SpeechToText"""
        # Handle common operations first
        if request.operation in ["status", "process"]:
            return await super().handle_api_request(request)

        # Handle component-specific operations
        try:
            if request.operation == "toggle_mock":
                self.use_mock = not self.use_mock
                return ComponentAPIResponse(
                    success=True,
                    message=f"Mock mode {'enabled' if self.use_mock else 'disabled'}",
                    component_type=self.__class__.__name__,
                    operation=request.operation,
                    data={"use_mock": self.use_mock}
                )
            elif request.operation == "clear_buffer":
                self.buffer = b''
                return ComponentAPIResponse(
                    success=True,
                    message="Buffer cleared",
                    component_type=self.__class__.__name__,
                    operation=request.operation
                )
            else:
                return ComponentAPIResponse(
                    success=False,
                    message=f"Unknown operation: {request.operation}",
                    component_type=self.__class__.__name__,
                    operation=request.operation
                )
        except Exception as e:
            logger.error(f"Error in API request: {str(e)}")
            return ComponentAPIResponse(
                success=False,
                message="Error processing API request",
                component_type=self.__class__.__name__,
                operation=request.operation,
                error=str(e)
            )