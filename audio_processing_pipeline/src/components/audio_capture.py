"""
Audio capture component for the pipeline.
"""

import asyncio
import numpy as np
import time
from typing import AsyncGenerator, Dict, Optional, Any
import wave
import os
from datetime import datetime
from pydantic import BaseModel
from loguru import logger

from src.core.component import Component, ComponentConfig

class AudioCaptureConfig(ComponentConfig):
    """Configuration for the AudioCapture component"""
    rate: int = 16000
    frames_per_buffer: int = 1024
    channels: int = 1
    format: int = 8  # 16-bit audio (pyaudio.paInt16)
    device_index: Optional[int] = None
    use_mock: bool = False
    export_audio: bool = False
    output_dir: str = "output/audio"

class AudioCapture(Component):
    """Component for capturing audio from microphone"""

    def __init__(self, config: AudioCaptureConfig):
        """Initialize the audio capture component"""
        super().__init__(config)
        self.config = config
        self.stream = None
        self.is_running = False
        self.audio_buffer = []
        self.pa = None  # PyAudio instance

        # Create output directory if exporting audio
        if self.config.export_audio:
            os.makedirs(self.config.output_dir, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the audio capture component"""
        logger.info("Initializing AudioCapture component...")

        # Initialize PyAudio if not using mock audio
        if not self.config.use_mock:
            try:
                import pyaudio
                self.pa = pyaudio.PyAudio()

                # Get the format value from PyAudio
                self.format_value = getattr(pyaudio, f"paInt{self.config.format * 2}")

                # List available audio devices to help with debugging
                logger.info("Available audio input devices:")
                for i in range(self.pa.get_device_count()):
                    dev_info = self.pa.get_device_info_by_index(i)
                    if dev_info.get('maxInputChannels') > 0:  # Only show input devices
                        logger.info(f"  Device {i}: {dev_info.get('name')}")

                # If device_index is None, try to find the default input device
                if self.config.device_index is None:
                    default_input = self.pa.get_default_input_device_info()
                    self.config.device_index = default_input.get('index')
                    logger.info(f"Using default input device: {default_input.get('name')} (index: {self.config.device_index})")

            except ImportError:
                logger.warning("PyAudio not installed. Falling back to mock audio.")
                self.config.use_mock = True
            except Exception as e:
                logger.error(f"Error initializing PyAudio: {str(e)}")
                logger.warning("Falling back to mock audio.")
                self.config.use_mock = True

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio data"""
        # This method is not used for audio capture as it uses a generator
        return data

    async def shutdown(self) -> None:
        """Shutdown the audio capture component"""
        logger.info("Shutting down AudioCapture component...")
        self.is_running = False

        # Close PyAudio stream if open
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {str(e)}")

        # Terminate PyAudio if initialized
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {str(e)}")

    async def audio_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio chunks from microphone"""
        if self.config.use_mock:
            # Generate mock audio for testing
            logger.info("Using mock audio generator")
            for _ in range(10):
                # Generate 1 second of mock audio (sine wave)
                t = np.linspace(0, 1, self.config.rate)
                audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

                # Reshape for stereo if needed
                if self.config.channels == 2:
                    audio_data = np.column_stack((audio_data, audio_data))

                # Export audio if enabled
                if self.config.export_audio:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{self.config.output_dir}/mock_audio_{timestamp}.npy"
                    np.save(filename, audio_data)

                yield audio_data
                await asyncio.sleep(1)
            return

        logger.info("Starting audio capture with PyAudio...")
        self.is_running = True

        try:
            import pyaudio

            # Open audio stream
            self.stream = self.pa.open(
                format=self.format_value,
                channels=self.config.channels,
                rate=self.config.rate,
                input=True,
                frames_per_buffer=self.config.frames_per_buffer,
                input_device_index=self.config.device_index
            )

            logger.info("Audio stream opened")
            logger.info(f"Capturing audio at {self.config.rate} Hz, {self.config.channels} channel(s)")

            # Read audio chunks
            while self.is_running:
                try:
                    # Read audio data
                    audio_bytes = self.stream.read(self.config.frames_per_buffer, exception_on_overflow=False)

                    # Convert to numpy array
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    # Reshape for stereo if needed
                    if self.config.channels == 2:
                        audio_data = audio_data.reshape(-1, 2)

                    # Export audio if enabled
                    if self.config.export_audio:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                        # Save as WAV file
                        filename = f"{self.config.output_dir}/audio_{timestamp}.wav"
                        with wave.open(filename, 'wb') as wf:
                            wf.setnchannels(self.config.channels)
                            wf.setsampwidth(2)  # 16-bit audio
                            wf.setframerate(self.config.rate)
                            # Convert back to int16 for WAV file
                            int16_data = (audio_data * 32768.0).astype(np.int16)
                            wf.writeframes(int16_data.tobytes())

                        # Also save as numpy array for easier processing
                        np_filename = f"{self.config.output_dir}/audio_{timestamp}.npy"
                        np.save(np_filename, audio_data)

                    yield audio_data

                    # Small delay to prevent CPU overload
                    await asyncio.sleep(0.001)

                except Exception as e:
                    logger.error(f"Error reading audio: {str(e)}")
                    await asyncio.sleep(0.1)  # Wait a bit before trying again

            # Close stream
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Audio stream closed")

        except Exception as e:
            logger.error(f"Error in audio capture: {str(e)}")
            # Fall back to mock audio
            logger.info("Falling back to mock audio")
            async for audio_data in self._mock_audio_generator():
                yield audio_data

        finally:
            self.is_running = False
            logger.info("Audio capture stopped")

    async def _mock_audio_generator(self) -> AsyncGenerator[np.ndarray, None]:
        """Generate mock audio data"""
        while self.is_running:
            # Generate 1 second of mock audio (sine wave)
            t = np.linspace(0, 1, self.config.rate)
            audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

            # Reshape for stereo if needed
            if self.config.channels == 2:
                audio_data = np.column_stack((audio_data, audio_data))

            # Export audio if enabled
            if self.config.export_audio:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{self.config.output_dir}/mock_audio_{timestamp}.npy"
                np.save(filename, audio_data)

            yield audio_data
            await asyncio.sleep(0.1)  # Simulate processing time


# """
# Audio capture component for the pipeline.
# """

# import asyncio
# import numpy as np
# import time
# from typing import AsyncGenerator, Dict, Optional, Any
# import wave
# import os
# from datetime import datetime
# from pydantic import BaseModel
# import sounddevice as sd
# from loguru import logger

# from src.core.component import Component, ComponentConfig

# class AudioCaptureConfig(ComponentConfig):
#     """Configuration for the AudioCapture component"""
#     rate: int = 16000
#     frames_per_buffer: int = 1024
#     channels: int = 1
#     format: int = 8  # 16-bit audio (pyaudio.paInt16)
#     device_index: Optional[int] = None
#     use_mock: bool = False
#     export_audio: bool = False

# class AudioCapture(Component):
#     """Component for capturing audio from microphone"""

#     def __init__(self, config: AudioCaptureConfig):
#         """Initialize the audio capture component"""
#         super().__init__(config)
#         self.config = config
#         self.stream = None
#         self.is_running = False
#         self.audio_buffer = []

#     async def initialize(self) -> None:
#         """Initialize the audio capture component"""
#         logger.info("Initializing AudioCapture component...")

#         # Create output directory if exporting audio
#         if self.config.export_audio:
#             os.makedirs("output/audio", exist_ok=True)

#     async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         """Process audio data"""
#         # This method is not used for audio capture as it uses a generator
#         return data

#     async def shutdown(self) -> None:
#         """Shutdown the audio capture component"""
#         logger.info("Shutting down AudioCapture component...")
#         self.is_running = False

#         if self.stream is not None:
#             self.stream.stop()
#             self.stream.close()
#             self.stream = None

#     def audio_callback(self, indata, frames, time_info, status):
#         """Callback function for audio stream"""
#         if status:
#             logger.warning(f"Audio status: {status}")

#         # Add audio data to buffer
#         self.audio_buffer.append(indata.copy())

#     async def audio_generator(self) -> AsyncGenerator[np.ndarray, None]:
#         """Generate audio chunks from microphone"""
#         if self.config.use_mock:
#             # Generate mock audio for testing
#             logger.info("Using mock audio generator")
#             for _ in range(10):
#                 # Generate 1 second of silence
#                 yield np.zeros((self.config.rate, self.config.channels), dtype=np.int16)
#                 await asyncio.sleep(1)
#             return

#         logger.info("Starting audio capture...")
#         self.is_running = True
#         self.audio_buffer = []

#         # Start audio stream
#         self.stream = sd.InputStream(
#             samplerate=self.config.rate,
#             channels=self.config.channels,
#             callback=self.audio_callback,
#             blocksize=self.config.frames_per_buffer
#         )
#         self.stream.start()

#         # Generate audio chunks
#         try:
#             start_time = time.time()

#             while self.is_running:
#                 # Wait for audio data
#                 if len(self.audio_buffer) > 0:
#                     # Get audio data from buffer
#                     audio_data = self.audio_buffer.pop(0)

#                     # Export audio if enabled
#                     if self.config.export_audio:
#                         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#                         filename = f"output/audio/chunk_{timestamp}.wav"
#                         with wave.open(filename, 'wb') as wf:
#                             wf.setnchannels(self.config.channels)
#                             wf.setsampwidth(2)  # 16-bit audio
#                             wf.setframerate(self.config.rate)
#                             wf.writeframes(audio_data.tobytes())

#                     # Yield audio data
#                     yield audio_data
#                 else:
#                     # Wait for more audio data
#                     await asyncio.sleep(0.01)

#         finally:
#             # Stop audio stream
#             if self.stream is not None:
#                 self.stream.stop()
#                 self.stream.close()
#                 self.stream = None

#             self.is_running = False
#             logger.info("Audio capture stopped")