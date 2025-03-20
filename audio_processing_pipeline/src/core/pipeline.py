import asyncio
import time
from contextlib import contextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from src.api.models import APIRequest, APIResponse, ComponentAPIRequest
from src.core.component import Component
from src.core.monitor import PipelineMonitor
from src.utils.logger import logger


class PipelineConfig:
    """Configuration for the audio pipeline"""
    def __init__(self, name, log_level="INFO", log_format="console",
                 components=None, api_enabled=True, api_auth_required=False,
                 api_key=None):
        self.name = name
        self.log_level = log_level
        self.log_format = log_format
        self.components = components or []
        self.api_enabled = api_enabled
        self.api_auth_required = api_auth_required
        self.api_key = api_key


class Pipeline:
    """Main pipeline that orchestrates component execution"""

    def __init__(self, config: PipelineConfig):
        self.name = config.name
        self.log_level = config.log_level
        self.components = []
        self.monitor = PipelineMonitor(self.name)
        self.api_enabled = config.api_enabled
        self.api_auth_required = config.api_auth_required
        self.api_key = config.api_key
        self._config = config

    def add_component(self, component: Component):
        """Add a component to the pipeline"""
        self.components.append(component)
        self.monitor.register_component(component.name)

    async def initialize(self):
        """Initialize all components"""
        logger.info(f"Initializing pipeline: {self.name}")
        for component in self.components:
            await component.initialize()

    async def shutdown(self):
        """Shutdown all components"""
        logger.info(f"Shutting down pipeline: {self.name}")
        for component in self.components:
            await component.shutdown()

    async def run(self,
                  input_generator: AsyncGenerator,
                  output_handler: Callable[[Any], None] = None,
                  max_iterations: int = None) -> None:
        """Run the pipeline"""
        # Initialize the pipeline
        await self.initialize()

        try:
            iteration = 0
            async for data in input_generator:
                if max_iterations is not None and iteration >= max_iterations:
                    break

                # Process the data through each component
                for i, component in enumerate(self.components):
                    with self.monitor.track(component.name):
                        data = await component.process(data)
                        # If data is None, skip the rest of the pipeline
                        if data is None:
                            break

                # Handle the output if provided
                if output_handler and data is not None:
                    await output_handler(data)

                iteration += 1
        finally:
            # Ensure we shut down properly
            await self.shutdown()

    async def handle_api_request(self, request: APIRequest) -> APIResponse:
        """Handle API requests to the pipeline"""
        if not self.api_enabled:
            return APIResponse(
                success=False,
                message="API access is disabled for this pipeline"
            )

        # Check authentication if required
        if self.api_auth_required:
            api_key = getattr(request, "api_key", None)
            if not api_key or api_key != self.api_key:
                return APIResponse(
                    success=False,
                    message="Authentication failed: Invalid API key"
                )

        # Handle component-specific requests
        if isinstance(request, ComponentAPIRequest):
            # Find the target component
            target_component = None
            for component in self.components:
                if component.__class__.__name__ == request.component_type:
                    target_component = component
                    break

            if target_component:
                return await target_component.handle_api_request(request)
            else:
                return APIResponse(
                    success=False,
                    message=f"Component not found: {request.component_type}"
                )

        # Handle pipeline-level requests
        try:
            # Example pipeline operations
            if hasattr(request, "operation"):
                operation = request.operation

                if operation == "status":
                    return APIResponse(
                        success=True,
                        message="Pipeline status retrieved",
                        data={
                            "name": self.name,
                            "components": [c.name for c in self.components],
                            "stats": self.monitor.get_stats()
                        }
                    )
                elif operation == "initialize":
                    await self.initialize()
                    return APIResponse(
                        success=True,
                        message="Pipeline initialized successfully"
                    )
                elif operation == "shutdown":
                    await self.shutdown()
                    return APIResponse(
                        success=True,
                        message="Pipeline shut down successfully"
                    )
                else:
                    return APIResponse(
                        success=False,
                        message=f"Unknown operation: {operation}"
                    )
            else:
                return APIResponse(
                    success=False,
                    message="Invalid request: operation not specified"
                )
        except Exception as e:
            logger.error(f"Error in API request: {str(e)}")
            return APIResponse(
                success=False,
                message="Error processing API request",
                error=str(e)
            )


class ConversationalPipeline(Pipeline):
    """Enhanced pipeline with conversational capabilities"""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.conversation_active = True
        self.response_ready = asyncio.Event()
        self.current_response = None

    async def run_conversation(self,
                          input_generator: AsyncGenerator,
                          max_turns: int = None) -> None:
        """Run a conversational pipeline"""
        # Initialize the pipeline
        await self.initialize()

        try:
            turn = 0
            while self.conversation_active:
                if max_turns is not None and turn >= max_turns:
                    break

                logger.info(f"Conversation turn {turn+1}")

                # Process input (speech to text)
                input_text = await self._process_input(input_generator)
                if not input_text:
                    # If no input detected, continue listening
                    continue

                logger.info(f"User: {input_text}")

                # Generate and speak response
                response_text, response_audio = await self._generate_response(input_text)
                if response_text:
                    logger.info(f"Assistant: {response_text}")

                turn += 1

                # Small pause between turns
                await asyncio.sleep(0.5)

        finally:
            # Ensure we shut down properly
            await self.shutdown()

    async def _process_input(self, input_generator) -> str:
        """Process audio input to get text"""
        # Find our speech-to-text component
        stt_component = None
        text_processing = None
        for component in self.components:
            if isinstance(component, SpeechToText):
                stt_component = component
            elif isinstance(component, TextProcessing):
                text_processing = component

        if not stt_component:
            logger.error("No speech-to-text component found")
            return None

        # Process a sequence of audio chunks until we get text
        timeout = time.time() + 10  # 10 second timeout for input
        buffer = ""

        async for audio_chunk in input_generator:
            # Reset timeout on voice activity
            is_speech = False

            # Run through the input part of the pipeline
            data = audio_chunk
            for component in self.components[:4]:  # Up to STT
                if isinstance(component, VAD):
                    is_speech, audio_data = await component.process(data)
                    if not is_speech:
                        data = None
                        break
                    data = (is_speech, audio_data)
                else:
                    data = await component.process(data)
                    if data is None:
                        break

            # If we got text from STT, accumulate it
            if isinstance(data, str) and data:
                buffer += " " + data
                timeout = time.time() + 3  # Reset timeout on speech

                # Check if we have a complete sentence
                if buffer.strip().endswith((".", "!", "?")) or len(buffer) > 100:
                    # Process the text with NLP if available
                    if text_processing:
                        processed = await text_processing.process(buffer.strip())
                        return processed["original_text"] if isinstance(processed, dict) else buffer.strip()
                    return buffer.strip()

            # Check timeout to prevent infinite waiting
            if time.time() > timeout:
                if buffer:
                    # Process whatever we have
                    if text_processing:
                        processed = await text_processing.process(buffer.strip())
                        return processed["original_text"] if isinstance(processed, dict) else buffer.strip()
                    return buffer.strip()
                return None

            # Small sleep to prevent CPU overuse
            await asyncio.sleep(0.01)

        return None

    async def _generate_response(self, input_text) -> Tuple[str, bytes]:
        """Generate and speak a response to the input"""
        # Find our response and TTS components
        response_generator = None
        tts_component = None

        for component in self.components:
            if isinstance(component, ResponseGenerator):
                response_generator = component
            elif isinstance(component, TextToSpeech):
                tts_component = component

        if not response_generator or not tts_component:
            logger.error("Missing response generator or TTS component")
            return None, None

        # Generate response text
        response_text = await response_generator.process(input_text)
        if not response_text:
            return None, None

        # Convert to speech
        response_audio = await tts_component.process(response_text)

        return response_text, response_audio