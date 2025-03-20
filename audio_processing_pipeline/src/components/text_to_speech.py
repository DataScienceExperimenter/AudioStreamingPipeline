# import asyncio
# import os
# import tempfile
# import traceback
# from typing import Dict, List, Optional, Union

# import numpy as np
# import torch
# from gtts import gTTS
# # for lightweight version
# import io

# from src.api.models import ComponentAPIRequest, ComponentAPIResponse
# from src.core.component import Component, ComponentConfig
# from src.utils.logger import logger


# class TextToSpeechConfig(ComponentConfig):
#     """Configuration for text-to-speech component"""
#     log_level: str = "INFO"
#     model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
#     device: str = "cpu"  # 'cpu' or 'cuda' for GPU if available
#     use_streaming: bool = True
#     sample_rate: int = 16000
#     pre_generate_common_phrases: bool = False
#     common_phrases_file: str = "common_phrases.json"
#     #for lightweight version
#     use_gtts: bool = False


# class TextToSpeech(Component):
#     """Text-to-speech component using TTS"""

#     def __init__(self, config: TextToSpeechConfig):
#         super().__init__(config)
#         self.model_name = config.model_name
#         self.device = config.device
#         self.use_streaming = config.use_streaming
#         self.sample_rate = config.sample_rate
#         self.pre_generate_common_phrases = config.pre_generate_common_phrases
#         self.common_phrases_file = config.common_phrases_file

#         self.model = None
#         self.vocoder = None
#         self.tts_pipeline = None
#         self.audio_cache = {}

#     async def initialize(self) -> None:
#         """Initialize the TTS model"""
#         logger.info(f"Initializing {self.name} with model {self.model_name}")

#         # Pre-generate common phrases if enabled
#         if self.pre_generate_common_phrases:
#             await self._load_common_phrases()

#         try:
#             # Initialize the TTS model in a separate thread
#             if not self.pre_generate_common_phrases:
#                 loop = asyncio.get_event_loop()
#                 await loop.run_in_executor(
#                     None,
#                     self._load_model
#                 )
#                 logger.info(f"TTS model {self.model_name} loaded successfully")
#             else:
#                 logger.info("Using pre-generated audio for common phrases")
#         except Exception as e:
#             logger.error(f"Failed to load TTS model: {str(e)}")
#             logger.error(traceback.format_exc())
#             # Fall back to simpler TTS
#             logger.warning("Falling back to gTTS for text-to-speech")

#         await super().initialize()

"""
Text-to-Speech component that converts text to audio.
Supports both ML-based TTS models and Google TTS for lightweight usage.
"""

import io
import asyncio
import numpy as np
import torch
from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field

from src.core.component import Component
from src.utils.logger import logger

# Import gTTS for lightweight TTS
from gtts import gTTS
from pydub import AudioSegment


class TextToSpeechConfig(BaseModel):
    """Configuration for the Text-to-Speech component"""
    name: str = "TextToSpeech"
    model_name: str = "facebook/mms-tts-eng"
    log_level: str = "INFO"
    device: str = "cpu"
    use_streaming: bool = True
    sample_rate: int = 16000
    use_gtts: bool = False  # Flag to use Google TTS instead of ML model
    voice: str = "en-US-Standard-B"  # Voice for gTTS
    cache_dir: Optional[str] = None
    max_text_length: int = 500
    chunk_size: int = 100  # Characters per chunk for long text



class TextToSpeech(Component):
    """
    Component that converts text to speech using either ML models or gTTS.

    For low-resource systems, set use_gtts=True to use Google's TTS service
    instead of loading large ML models.
    """

    def __init__(self, config: TextToSpeechConfig):
        """Initialize the Text-to-Speech component"""
        super().__init__(config)
        self.config = config
        self.model = None
        self.processor = None
        self.device = torch.device(config.device)
        self.use_gtts = config.use_gtts
        self.voice = config.voice
        self.sample_rate = config.sample_rate
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the TTS model"""
        try:
            if not self.use_gtts:
                # Only load ML models if not using gTTS
                logger.info(f"Loading TTS model: {self.config.model_name}")

                # Import here to avoid loading transformers if using gTTS
                from transformers import AutoProcessor, AutoModel

                # Load processor and model
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir
                )

                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir
                ).to(self.device)

                logger.info(f"TTS model loaded on {self.device}")
            else:
                logger.info("Using Google TTS for lightweight operation")

            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TTS component: {str(e)}")
            return False

    async def process(self, text: str) -> Optional[bytes]:
        """
        Convert text to speech audio.

        Args:
            text: The text to convert to speech

        Returns:
            Audio data as bytes or None if processing failed
        """
        if not self.is_initialized:
            logger.error("TTS component not initialized")
            return None

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text input: {text}")
            return None

        try:
            # Truncate text if too long
            if len(text) > self.config.max_text_length:
                logger.warning(f"Text too long ({len(text)} chars), truncating to {self.config.max_text_length}")
                text = text[:self.config.max_text_length]

            if self.use_gtts:
                return await self._process_with_gtts(text)
            else:
                return await self._process_with_ml_model(text)

        except Exception as e:
            logger.error(f"Error in TTS processing: {str(e)}")
            return None

    async def _process_with_gtts(self, text: str) -> Optional[bytes]:
        """Process text using Google TTS"""
        try:
            # Run gTTS in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                None, self._generate_gtts_audio, text
            )
            return audio_bytes
        except Exception as e:
            logger.error(f"Error in gTTS processing: {str(e)}")
            return None

    def _generate_gtts_audio(self, text: str) -> bytes:
        """Generate audio using gTTS (runs in thread pool)"""
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)

        # Save to BytesIO object
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # Convert MP3 to WAV using pydub
        audio = AudioSegment.from_mp3(mp3_fp)

        # Resample to match our sample rate
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)

        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Export as WAV
        wav_fp = io.BytesIO()
        audio.export(wav_fp, format="wav")
        wav_fp.seek(0)

        # Return the WAV data
        return wav_fp.read()

    async def _process_with_ml_model(self, text: str) -> Optional[bytes]:
        """Process text using ML model"""
        try:
            # For long text, split into chunks and process separately
            if len(text) > self.config.chunk_size:
                return await self._process_long_text(text)

            # Process with ML model
            with torch.no_grad():
                inputs = self.processor(
                    text=text,
                    return_tensors="pt"
                ).to(self.device)

                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=None,
                    vocoder=None
                )

            # Convert to bytes
            audio_np = speech.cpu().numpy()
            audio_bytes = self._numpy_to_wav(audio_np)
            return audio_bytes

        except Exception as e:
            logger.error(f"Error in ML model processing: {str(e)}")
            return None

    async def _process_long_text(self, text: str) -> Optional[bytes]:
        """Process long text by splitting into chunks"""
        # Split text into sentences or chunks
        chunks = self._split_text(text, self.config.chunk_size)

        # Process each chunk
        audio_segments = []
        for chunk in chunks:
            if not chunk.strip():
                continue

            chunk_audio = await self._process_with_ml_model(chunk)
            if chunk_audio:
                # Convert bytes to AudioSegment
                chunk_fp = io.BytesIO(chunk_audio)
                segment = AudioSegment.from_wav(chunk_fp)
                audio_segments.append(segment)

        if not audio_segments:
            return None

        # Combine all segments
        combined = audio_segments[0]
        for segment in audio_segments[1:]:
            combined += segment

        # Export as WAV
        wav_fp = io.BytesIO()
        combined.export(wav_fp, format="wav")
        wav_fp.seek(0)
        return wav_fp.read()

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks, trying to split at sentence boundaries"""
        # Try to split at sentence boundaries
        sentences = []
        current_sentence = ""

        # Simple sentence splitting
        for char in text:
            current_sentence += char
            if char in ['.', '!', '?'] and len(current_sentence) > 0:
                sentences.append(current_sentence)
                current_sentence = ""

        # Add any remaining text
        if current_sentence:
            sentences.append(current_sentence)

        # Combine short sentences to reach chunk_size
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _numpy_to_wav(self, audio_np: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes"""
        import wave
        import struct

        # Normalize audio to 16-bit range
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_np = (audio_np * 32767.0).astype(np.int16)

        # Create WAV file in memory
        wav_fp = io.BytesIO()
        with wave.open(wav_fp, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            # Write audio data
            for sample in audio_np:
                wav_file.writeframes(struct.pack('<h', sample))

        wav_fp.seek(0)
        return wav_fp.read()

    async def shutdown(self) -> bool:
        """Shutdown the TTS component"""
        try:
            # Clear model from memory
            if self.model is not None:
                del self.model
                self.model = None

            if self.processor is not None:
                del self.processor
                self.processor = None

            # Force garbage collection
            import gc
            gc.collect()

            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()

            self.is_initialized = False
            return True
        except Exception as e:
            logger.error(f"Error shutting down TTS component: {str(e)}")
            return False