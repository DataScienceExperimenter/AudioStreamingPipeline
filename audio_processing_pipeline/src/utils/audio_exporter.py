import base64
import os
import wave
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display

from src.utils.logger import logger


class AudioExporter:
    """Utility class for exporting and visualizing audio data"""

    @staticmethod
    def save_wav(audio_data: bytes, filename: str, channels: int = 1,
                 sample_width: int = 2, framerate: int = 16000) -> str:
        """Save audio data to a WAV file

        Args:
            audio_data: Raw audio bytes
            filename: Output filename
            channels: Number of audio channels
            sample_width: Sample width in bytes (2 for 16-bit)
            framerate: Sample rate in Hz

        Returns:
            The full path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)
        
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(framerate)
            wf.writeframes(audio_data)

        logger.info(f"Saved audio to {filepath}")
        return filepath

    @staticmethod
    def plot_audio(audio_data: bytes, title: str = "Audio Waveform", 
                   save_path: Optional[str] = None) -> None:
        """Plot the audio waveform

        Args:
            audio_data: Raw audio bytes
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        plt.figure(figsize=(10, 4))
        plt.plot(audio_array)
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved audio plot to {save_path}")
        else:
            plt.show()

    @staticmethod
    def play_audio(audio_data: bytes, rate: int = 16000) -> None:
        """Play audio in notebook

        Args:
            audio_data: Raw audio bytes
            rate: Sample rate in Hz
        """
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Normalize to -1.0 to 1.0 range for Audio display
        audio_normalized = audio_array.astype(np.float32) / 32767.0
        display(Audio(audio_normalized, rate=rate))

    @staticmethod
    def bytes_to_base64(audio_data: bytes) -> str:
        """Convert audio bytes to base64 string for API responses

        Args:
            audio_data: Raw audio bytes

        Returns:
            Base64 encoded string
        """
        return base64.b64encode(audio_data).decode('utf-8')

    @staticmethod
    def base64_to_bytes(base64_str: str) -> bytes:
        """Convert base64 string to audio bytes

        Args:
            base64_str: Base64 encoded string

        Returns:
            Raw audio bytes
        """
        return base64.b64decode(base64_str)