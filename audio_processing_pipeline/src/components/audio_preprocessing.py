from typing import Tuple, Union

import numpy as np
import librosa

from src.api.models import ComponentAPIRequest, ComponentAPIResponse
from src.core.component import Component, ComponentConfig
from src.utils.logger import logger


class AudioPreprocessingConfig(ComponentConfig):
    """Configuration for audio preprocessing"""
    apply_gain: bool = True
    gain_factor: float = 1.0
    apply_noise_reduction: bool = False
    apply_normalization: bool = False
    target_sample_rate: int = 16000


class AudioPreprocessing(Component):
    """Audio preprocessing component"""

    def __init__(self, config: AudioPreprocessingConfig):
        super().__init__(config)
        self.apply_gain = config.apply_gain
        self.gain_factor = config.gain_factor
        self.apply_noise_reduction = config.apply_noise_reduction
        self.apply_normalization = config.apply_normalization
        self.target_sample_rate = config.target_sample_rate

    async def process(self, data: Union[bytes, Tuple[bool, bytes]]) -> bytes:
        """Process audio data

        Args:
            data: Either raw audio bytes or (is_speech, audio_bytes) from VAD

        Returns:
            Processed audio bytes
        """
        if not self.initialized:
            raise RuntimeError("Component not initialized")

        # Extract audio data from VAD output if needed
        if isinstance(data, tuple) and len(data) == 2:
            is_speech, audio_data = data
            # Skip processing if no speech detected
            if not is_speech:
                return None
        else:
            audio_data = data

        # Convert to numpy for processing
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

        # Apply gain if enabled
        if self.apply_gain:
            audio_array = audio_array * self.gain_factor

        # Apply noise reduction if enabled
        if self.apply_noise_reduction:
            # Simple noise reduction by spectral subtraction
            # In a real implementation, you would use a more sophisticated algorithm
            # This is a simplified version for demonstration
            fft_size = 512
            hop_length = fft_size // 4

            # Compute spectrogram
            stft = librosa.stft(audio_array / 32767.0, n_fft=fft_size, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Estimate noise from the first few frames
            noise_estimate = np.mean(magnitude[:, :5], axis=1, keepdims=True)

            # Subtract noise (with flooring to avoid negative values)
            magnitude = np.maximum(magnitude - noise_estimate * 2, 0)

            # Reconstruct signal
            stft_denoised = magnitude * np.exp(1j * phase)
            audio_array = librosa.istft(stft_denoised, hop_length=hop_length) * 32767.0

        # Apply normalization if enabled
        if self.apply_normalization:
            # Normalize to use full dynamic range
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 32767.0

        # Clip to prevent overflow
        audio_array = np.clip(audio_array, -32768, 32767)

        # Convert back to bytes
        processed_audio = audio_array.astype(np.int16).tobytes()

        return processed_audio

    async def handle_api_request(self, request: ComponentAPIRequest) -> ComponentAPIResponse:
        """Handle API requests specific to AudioPreprocessing"""
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
                if "apply_gain" in request.config:
                    self.apply_gain = bool(request.config["apply_gain"])
                if "gain_factor" in request.config:
                    self.gain_factor = float(request.config["gain_factor"])
                if "apply_noise_reduction" in request.config:
                    self.apply_noise_reduction = bool(request.config["apply_noise_reduction"])
                if "apply_normalization" in request.config:
                    self.apply_normalization = bool(request.config["apply_normalization"])
                if "target_sample_rate" in request.config:
                    self.target_sample_rate = int(request.config["target_sample_rate"])

                return ComponentAPIResponse(
                    success=True,
                    message="Configuration updated successfully",
                    component_type=self.__class__.__name__,
                    operation=request.operation,
                    data={
                        "apply_gain": self.apply_gain,
                        "gain_factor": self.gain_factor,
                        "apply_noise_reduction": self.apply_noise_reduction,
                        "apply_normalization": self.apply_normalization,
                        "target_sample_rate": self.target_sample_rate
                    }
                )
            elif request.operation == "get_config":
                return ComponentAPIResponse(
                    success=True,
                    message="Configuration retrieved successfully",
                    component_type=self.__class__.__name__,
                    operation=request.operation,
                    data={
                        "apply_gain": self.apply_gain,
                        "gain_factor": self.gain_factor,
                        "apply_noise_reduction": self.apply_noise_reduction,
                        "apply_normalization": self.apply_normalization,
                        "target_sample_rate": self.target_sample_rate
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