import os
import gc
import numpy as np
from typing import Dict, Any

# Try to import logger, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("speech_to_text")

class SpeechToText:
    def __init__(self, config=None, performance_tracker=None):
        self.name = "SpeechToText"
        self.performance_tracker = performance_tracker
        self.model_name = config.get("model_name", "facebook/wav2vec2-base-960h") if config else "facebook/wav2vec2-base-960h"
        self.use_mock = config.get("use_mock", False) if config else False
        self.model = None
        self.processor = None

        # Try to use whisper if available
        self.use_whisper = config.get("use_whisper", True) if config else True
        self.whisper_model = None

        logger.info(f"Initializing {self.name}")

    async def initialize(self):
        try:
            if not self.use_mock:
                # Try to use whisper first
                if self.use_whisper:
                    try:
                        import whisper
                        self.whisper_model = whisper.load_model("base")
                        logger.info("Loaded Whisper model: base")
                        return True
                    except ImportError:
                        logger.warning("Whisper not installed. Trying transformers.")
                        self.use_whisper = False

                # Try to import transformers as fallback
                try:
                    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                    # Load model and processor
                    self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                    self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)

                    logger.info(f"Loaded Wav2Vec2 model: {self.model_name}")
                except ImportError:
                    logger.warning("Transformers not installed. Using SpeechRecognition.")
                    try:
                        import speech_recognition as sr
                        self.recognizer = sr.Recognizer()
                        logger.info("Using SpeechRecognition")
                    except ImportError:
                        logger.warning("SpeechRecognition not installed. Using mock STT.")
                        self.use_mock = True

            logger.info(f"{self.name} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing {self.name}: {str(e)}")
            self.use_mock = True
            return False

    async def process(self, audio_data):
        """Convert audio to text"""
        if self.performance_tracker:
            self.performance_tracker.start_component("SpeechToText")

        try:
            # Check if audio data is empty
            if len(audio_data) == 0:
                if self.performance_tracker:
                    self.performance_tracker.end_component("SpeechToText")
                return ""

            # Use mock STT if enabled or if model failed to load
            if self.use_mock:
                # Return a fixed transcription for testing
                transcription = "This is a test of the audio processing pipeline."
            elif self.use_whisper and self.whisper_model:
                # Use whisper for transcription
                try:
                    # Save audio to a temporary file
                    temp_file = "temp_audio.wav"
                    import soundfile as sf
                    sf.write(temp_file, audio_data, 16000)

                    # Transcribe with whisper
                    result = self.whisper_model.transcribe(temp_file)
                    transcription = result["text"].strip()

                    # Remove temporary file
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Error in Whisper transcription: {str(e)}")
                    transcription = "Error transcribing audio."
            elif hasattr(self, 'recognizer'):
                # Use SpeechRecognition
                try:
                    import speech_recognition as sr
                    import io
                    import wave

                    # Convert audio data to WAV format
                    byte_io = io.BytesIO()
                    with wave.open(byte_io, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 2 bytes for 16-bit audio
                        wf.setframerate(16000)
                        wf.writeframes((audio_data * 32768).astype(np.int16).tobytes())

                    # Create AudioData object
                    audio_data_obj = sr.AudioData(byte_io.getvalue(), 16000, 2)

                    # Recognize speech using Google Speech Recognition
                    transcription = self.recognizer.recognize_google(audio_data_obj)
                except Exception as e:
                    logger.error(f"Error in SpeechRecognition: {str(e)}")
                    transcription = "Error transcribing audio."
            else:
                # Use the transformers model for transcription
                try:
                    import torch

                    # Ensure audio data is in the correct format
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)

                    # Process audio with the model
                    inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        logits = self.model(inputs.input_values).logits

                    # Decode the model output
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.processor.batch_decode(predicted_ids)[0]

                except Exception as e:
                    logger.error(f"Error in STT processing: {str(e)}")
                    transcription = "Error transcribing audio."

            if self.performance_tracker:
                elapsed = self.performance_tracker.end_component("SpeechToText")
                logger.info(f"SpeechToText processing time: {elapsed:.3f}s")

            # Log the transcription with a special format for visibility
            print("\n" + "=" * 50)
            print("🎤 TRANSCRIPTION RESULT")
            print("=" * 50)
            print(f"\"{transcription}\"")
            print("=" * 50 + "\n")

            logger.info(f"🎤 TRANSCRIPTION: \"{transcription}\"")
            return transcription

        except Exception as e:
            logger.error(f"Error in SpeechToText: {str(e)}")

            if self.performance_tracker:
                self.performance_tracker.end_component("SpeechToText")

            return "Error transcribing audio."

    async def shutdown(self):
        # Clear model from memory
        self.model = None
        self.processor = None
        self.whisper_model = None

        # Force garbage collection
        gc.collect()

        logger.info(f"{self.name} shutdown successfully")
        return True
