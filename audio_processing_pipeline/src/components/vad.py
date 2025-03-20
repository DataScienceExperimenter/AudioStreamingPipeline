from typing import Tuple

import numpy as np

from src.api.models import ComponentAPIRequest, ComponentAPIResponse
from src.core.component import Component, ComponentConfig
from src.utils.logger import logger


class VADConfig(ComponentConfig):
    """Configuration for Voice Activity Detection"""
    energy_threshold: float = 0.01
    window_size: int = 4
    speech_threshold: float = 0.5
    advanced_mode: bool = False  # Use more sophisticated VAD when True


class VAD(Component):
    """Voice Activity Detection component"""

    def __init__(self, config: VADConfig):
        super().__init__(config)
        self.energy_threshold = config.energy_threshold
        self.window_size = config.window_size
        self.speech_threshold = config.speech_threshold
        self.advanced_mode = config.advanced_mode
        self.energy_history = []

    async def process(self, data: bytes) -> Tuple[bool, bytes]:
        """Detect voice activity in audio data

        Returns:
            Tuple of (is_speech, audio_data)
        """
        if not self.initialized:
            raise RuntimeError("Component not initialized")

        # Convert bytes to numpy array
        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)

        if self.advanced_mode:
            # More sophisticated VAD using energy and zero-crossing rate
            energy = np.mean(np.abs(audio_array)) / 32767.0

            # Calculate zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_array)))) / len(audio_array)

            # Update energy history
            self.energy_history.append(energy)
            if len(self.energy_history) > self.window_size:
                self.energy_history.pop(0)

            # Calculate average energy over window
            avg_energy = np.mean(self.energy_history)

            # Combine energy and zero-crossing rate for decision
            # High energy and moderate zero-crossing rate typically indicates speech
            is_speech = (avg_energy > self.energy_threshold and
                         0.01 < zero_crossings < 0.15)
        else:
            # Simple energy-based detection
            energy = np.mean(np.abs(audio_array)) / 32767.0
            is_speech = energy > self.energy_threshold

        return is_speech, data

    async def handle_api_request(self, request: ComponentAPIRequest) -> ComponentAPIResponse:
        """Handle API requests specific to VAD"""
        # Handle common operations first
        if request.operation in ["status", "process"]:
            return await super().handle_api_request(request)

        # Handle component-specific operations
        try:
            if request.operation == "update_config":
                if not request.config:
                    return ComponentAPIResponse(
                        success=False,
                        message="No configuration provided",
                        component_type=self.__class__.__name__,
                        operation=request.operation
                    )

                # Update configuration
                if "energy_threshold" in request.config:
                    self.energy_threshold = float(request.config["energy_threshold"])
                if "window_size" in request.config:
                    self.window_size = int(request.config["window_size"])
                if "speech_threshold" in request.config:
                    self.speech_threshold = float(request.config["speech_threshold"])
                if "advanced_mode" in request.config:
                    self.advanced_mode = bool(request.config["advanced_mode"])

                return ComponentAPIResponse(
                    success=True,
                    message="Configuration updated successfully",
                    component_type=self.__class__.__name__,
                    operation=request.operation,
                    data={
                        "energy_threshold": self.energy_threshold,
                        "window_size": self.window_size,
                        "speech_threshold": self.speech_threshold,
                        "advanced_mode": self.advanced_mode
                    }
                )
            elif request.operation == "get_config":
                return ComponentAPIResponse(
                    success=True,
                    message="Configuration retrieved successfully",
                    component_type=self.__class__.__name__,
                    operation=request.operation,
                    data={
                        "energy_threshold": self.energy_threshold,
                        "window_size": self.window_size,
                        "speech_threshold": self.speech_threshold,
                        "advanced_mode": self.advanced_mode
                    }
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