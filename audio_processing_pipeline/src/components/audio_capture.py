import os
import time
import numpy as np
from typing import Dict, Any
import asyncio

# Try to import logger, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("audio_capture")

class AudioCapture:
    def __init__(self, config=None, performance_tracker=None):
        self.name = "AudioCapture"
        self.performance_tracker = performance_tracker
        self.rate = config.get("rate", 16000) if config else 16000
        self.frames_per_buffer = config.get("frames_per_buffer", 1024) if config else 1024
        self.channels = config.get("channels", 1) if config else 1
        self.format = config.get("format", 8) if config else 8  # 16-bit audio
        self.device_index = config.get("device_index") if config else None
        self.export_audio = config.get("export_audio", True) if config else True
        self.output_dir = config.get("output_dir", "output/audio") if config else "output/audio"
        self.is_running = False
        self.stream = None
        self.pyaudio_instance = None

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Initializing {self.name}")

    async def initialize(self):
        try:
            import pyaudio
            self.pyaudio_instance = pyaudio.PyAudio()
            logger.info(f"{self.name} initialized successfully")
            return True
        except ImportError:
            logger.error("PyAudio not installed. Please install it with 'pip install pyaudio'")
            return False
        except Exception as e:
            logger.error(f"Error initializing {self.name}: {str(e)}")
            return False

    async def capture_audio(self, duration):
        """Capture audio for the specified duration"""
        if self.performance_tracker:
            self.performance_tracker.start_component("AudioCapture_recording")

        logger.info(f"Capturing audio for {duration} seconds...")

        try:
            # Use PyAudio directly for more reliable audio capture
            import pyaudio

            if not self.pyaudio_instance:
                self.pyaudio_instance = pyaudio.PyAudio()

            stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )

            # Calculate how many frames to read
            frames_to_read = int(16000 * duration / 1024)

            # Read audio data
            frames = []
            for i in range(frames_to_read):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)

            # Close the stream
            stream.stop_stream()
            stream.close()

            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

            # Save the audio file for debugging
            if self.export_audio:
                import wave
                filename = os.path.join(self.output_dir, f"recording_{int(time.time())}.wav")
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 2 bytes for 16-bit audio
                    wf.setframerate(16000)
                    wf.writeframes(b''.join(frames))
                logger.info(f"Saved audio to {filename}")

            if self.performance_tracker:
                elapsed = self.performance_tracker.end_component("AudioCapture_recording")
                logger.info(f"Audio recording time: {elapsed:.3f}s")

            return audio_data

        except Exception as e:
            logger.error(f"Error capturing audio: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            if self.performance_tracker:
                self.performance_tracker.end_component("AudioCapture_recording")

            # Return empty array on error
            return np.array([], dtype=np.float32)

    async def process(self, data):
        """Process input data (not used for AudioCapture)"""
        if self.performance_tracker:
            self.performance_tracker.start_component("AudioCapture")

        # AudioCapture doesn't process input data
        result = data

        if self.performance_tracker:
            elapsed = self.performance_tracker.end_component("AudioCapture")
            logger.info(f"AudioCapture processing time: {elapsed:.3f}s")

        return result

    async def shutdown(self):
        """Shutdown the audio capture component"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()

        logger.info(f"{self.name} shutdown successfully")
        return True
