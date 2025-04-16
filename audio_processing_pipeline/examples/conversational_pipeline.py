import asyncio
import os
import gc
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, Any

# Suppress all warnings
warnings.filterwarnings("ignore")

# Configure logger
try:
    from loguru import logger
    logger.remove()
    logger.add(lambda msg: print(msg, flush=True), level="INFO")
    logger.add("logs/pipeline.log", rotation="10 MB", level="DEBUG", enqueue=True)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pipeline")
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from src.components.audio_capture import AudioCapture
from src.components.vad import VAD
from src.components.audio_preprocessing import AudioPreprocessing
from src.components.speech_to_text import SpeechToText
from src.components.text_processing import TextProcessing
from src.components.response_generator import ResponseGenerator
from src.components.text_to_speech import TextToSpeech

# Performance metrics tracker
class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            "total_time": 0,
            "components": {},
            "start_time": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        }
        self.start_times = {}
        self.total_start_time = None

    def start_pipeline(self):
        self.total_start_time = time.time()
        self.metrics = {
            "total_time": 0,
            "components": {},
            "start_time": datetime.now().strftime("%H:%M:%S.%f")[:-3]
        }

    def start_component(self, component_name):
        self.start_times[component_name] = time.time()

    def end_component(self, component_name):
        if component_name in self.start_times:
            elapsed = time.time() - self.start_times[component_name]
            if component_name not in self.metrics["components"]:
                self.metrics["components"][component_name] = []
            self.metrics["components"][component_name].append(elapsed)
            return elapsed
        return 0

    def end_pipeline(self):
        if self.total_start_time:
            self.metrics["total_time"] = time.time() - self.total_start_time
            self.metrics["end_time"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            return self.metrics["total_time"]
        return 0

    def print_summary(self):
        summary = {
            "total_time": f"{self.metrics['total_time']:.3f}s",
            "start_time": self.metrics["start_time"],
            "end_time": self.metrics.get("end_time", "N/A"),
            "components": {}
        }

        for component, times in self.metrics["components"].items():
            avg_time = sum(times) / len(times) if times else 0
            summary["components"][component] = f"{avg_time:.3f}s"

        print("\n" + "=" * 50)
        print("📊 PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Start time: {summary['start_time']}")
        print(f"End time: {summary['end_time']}")
        print(f"Total processing time: {summary['total_time']}")
        print("\nComponent times:")

        for component, time_str in summary["components"].items():
            print(f"  - {component}: {time_str}")

        print("=" * 50 + "\n")

# Conversational Pipeline
class ConversationalPipeline:
    def __init__(self, performance_tracker=None):
        self.name = "ConversationalPipeline"
        self.performance_tracker = performance_tracker
        self.components = []

        logger.info(f"Initializing {self.name}")

    def add_component(self, component):
        """Add a component to the pipeline"""
        self.components.append(component)
        logger.info(f"Added component: {component.name}")

    async def initialize(self):
        """Initialize all components in the pipeline"""
        for component in self.components:
            try:
                await component.initialize()
            except Exception as e:
                logger.error(f"Error initializing component {component.name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False

        logger.info(f"{self.name} initialized successfully")
        return True

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through all components in the pipeline"""
        # Start tracking pipeline performance
        if self.performance_tracker:
            self.performance_tracker.start_pipeline()

        result = {}

        # Extract audio data from the input dictionary
        audio_data = data.get("audio_data")
        if audio_data is None:
            logger.error("No audio data provided")
            if self.performance_tracker:
                self.performance_tracker.end_pipeline()
            return result

        try:
            # Process through components
            current_data = audio_data

            # VAD (component 0)
            if len(self.components) > 0:
                vad_component = self.components[0]
                has_speech, processed_audio = await vad_component.process(current_data)
                result["has_speech"] = has_speech

                if not has_speech:
                    logger.info("No speech detected")
                    if self.performance_tracker:
                        self.performance_tracker.end_pipeline()
                    return result

                current_data = processed_audio

            # AudioPreprocessing (component 1)
            if len(self.components) > 1:
                preprocessing_component = self.components[1]
                processed_audio = await preprocessing_component.process(current_data)
                current_data = processed_audio

            # SpeechToText (component 2)
            if len(self.components) > 2:
                stt_component = self.components[2]
                transcription = await stt_component.process(current_data)
                result["transcription"] = transcription

                if not transcription:
                    logger.info("No transcription generated")
                    if self.performance_tracker:
                        self.performance_tracker.end_pipeline()
                    return result

                current_data = transcription

            # TextProcessing (component 3)
            if len(self.components) > 3:
                text_processing_component = self.components[3]
                processed_text = await text_processing_component.process(current_data)

                if isinstance(processed_text, dict):
                    result.update(processed_text)
                else:
                    result["processed_text"] = processed_text

                current_data = result

            # ResponseGenerator (component 4)
            if len(self.components) > 4:
                response_generator_component = self.components[4]
                response_data = await response_generator_component.process(current_data)

                if isinstance(response_data, dict) and "response" in response_data:
                    result["response"] = response_data["response"]
                    for key, value in response_data.items():
                        if key != "response":
                            result[key] = value
                else:
                    result["response"] = response_data

                current_data = result["response"]

            # TextToSpeech (component 5)
            if len(self.components) > 5:
                tts_component = self.components[5]

                if not isinstance(current_data, str):
                    if isinstance(current_data, dict) and "response" in current_data:
                        current_data = current_data["response"]
                    else:
                        current_data = str(current_data)

                speech_file = await tts_component.process(current_data)
                result["speech_file"] = speech_file

            # End tracking pipeline performance
            if self.performance_tracker:
                total_time = self.performance_tracker.end_pipeline()
                logger.info(f"Total pipeline processing time: {total_time:.3f}s")

            return result

        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            if self.performance_tracker:
                self.performance_tracker.end_pipeline()

            return result

    async def shutdown(self):
        """Shutdown all components in the pipeline"""
        for component in self.components:
            try:
                await component.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down component {component.name}: {str(e)}")

        logger.info(f"{self.name} shutdown successfully")
        return True

async def play_prompt_sound(sound_type: str):
    """Play a sound to indicate the system is ready to listen or has finished listening"""
    try:
        # Check if pygame is installed
        try:
            import pygame
            pygame.mixer.init()

            # Define sound files
            sound_files = {
                "start": "assets/start_listening.wav",
                "stop": "assets/stop_listening.wav"
            }

            # Create assets directory if it doesn't exist
            os.makedirs("assets", exist_ok=True)

            # Check if sound file exists
            sound_file = sound_files.get(sound_type)
            if sound_file and os.path.exists(sound_file):
                # Play sound
                pygame.mixer.Sound(sound_file).play()
                await asyncio.sleep(0.5)  # Wait for sound to play
            else:
                # Generate a simple beep using pygame
                pygame.mixer.quit()
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

                if sound_type == "start":
                    # High-pitched beep for start
                    pygame.mixer.Sound(pygame.sndarray.make_sound(
                        np.sin(2 * np.pi * np.arange(44100 * 0.5) * 880 / 44100).astype(np.float32)
                    )).play()
                else:
                    # Low-pitched beep for stop
                    pygame.mixer.Sound(pygame.sndarray.make_sound(
                        np.sin(2 * np.pi * np.arange(44100 * 0.5) * 440 / 44100).astype(np.float32)
                    )).play()

                await asyncio.sleep(0.5)  # Wait for sound to play

        except ImportError:
            # If pygame is not installed, use a simple print message
            if sound_type == "start":
                print("\a")  # Terminal bell
                logger.info("🔊 BEEP! Start speaking now.")
            else:
                print("\a")  # Terminal bell
                logger.info("🔊 BEEP! Finished listening.")

    except Exception as e:
        logger.error(f"Error playing prompt sound: {str(e)}")

async def play_audio_file(audio_file: str):
    """Play an audio file"""
    try:
        # Try to play the response using pygame
        import pygame
        pygame.mixer.init()

        if audio_file.endswith(".mp3"):
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # Wait for the response to finish playing
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
        elif audio_file.endswith(".aiff") or audio_file.endswith(".wav"):
            sound = pygame.mixer.Sound(audio_file)
            sound.play()

            # Wait for the sound to finish playing
            duration = sound.get_length()
            await asyncio.sleep(duration)
    except ImportError:
        logger.info(f"Response saved to {audio_file}")
    except Exception as e:
        logger.error(f"Error playing audio: {str(e)}")

        # Try using macOS afplay as a fallback
        try:
            import subprocess
            subprocess.run(["afplay", audio_file], check=True)
        except Exception:
            logger.info(f"Response saved to {audio_file}")

async def run_conversational_pipeline():
    """Run the conversational pipeline with real microphone input"""
    # Create output directories
    os.makedirs("output/audio", exist_ok=True)
    os.makedirs("output/tts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create performance tracker
    performance_tracker = PerformanceTracker()

    # Create pipeline components
    audio_capture = AudioCapture({"rate": 16000, "frames_per_buffer": 1024}, performance_tracker)
    vad = VAD({"energy_threshold": 0.01, "window_size": 4}, performance_tracker)
    audio_preprocessing = AudioPreprocessing({"apply_gain": True, "gain_factor": 1.5}, performance_tracker)

    # Try to use whisper for better speech recognition if available
    try:
        import whisper
        speech_to_text = SpeechToText({"use_whisper": True}, performance_tracker)
        logger.info("Using Whisper for speech recognition")
    except ImportError:
        # Try to use SpeechRecognition as fallback
        try:
            import speech_recognition
            speech_to_text = SpeechToText({"use_mock": False}, performance_tracker)
            logger.info("Using SpeechRecognition for speech recognition")
        except ImportError:
            speech_to_text = SpeechToText({"use_mock": True}, performance_tracker)
            logger.info("Using mock speech recognition")

    text_processing = TextProcessing({"use_mock": True}, performance_tracker)

    # Try to use transformers for better response generation
    try:
        from transformers import pipeline
        response_generator = ResponseGenerator({"use_transformer": True}, performance_tracker)
        logger.info("Using transformer model for response generation")
    except ImportError:
        response_generator = ResponseGenerator({"use_transformer": False}, performance_tracker)
        logger.info("Using rule-based response generation")

    # Try to use gTTS for text-to-speech
    try:
        from gtts import gTTS
        text_to_speech = TextToSpeech({"use_gtts": True}, performance_tracker)
        logger.info("Using gTTS for text-to-speech")
    except ImportError:
        text_to_speech = TextToSpeech({"use_gtts": False}, performance_tracker)
        logger.info("Using mock text-to-speech")

    # Create pipeline
    pipeline = ConversationalPipeline(performance_tracker)

    # Add components to pipeline
    pipeline.add_component(vad)
    pipeline.add_component(audio_preprocessing)
    pipeline.add_component(speech_to_text)
    pipeline.add_component(text_processing)
    pipeline.add_component(response_generator)
    pipeline.add_component(text_to_speech)

    try:
        # Initialize components
        await audio_capture.initialize()
        await pipeline.initialize()

        # Print welcome message
        print("\n" + "=" * 50)
        print("🎤 CONVERSATIONAL AUDIO PIPELINE")
        print("=" * 50)
        print("This pipeline captures audio from your microphone,")
        print("processes it, and generates a spoken response.")
        print("Press Ctrl+C to exit.")
        print("=" * 50 + "\n")

        # Run the conversation loop
        continue_conversation = True
        for turn in range(1, 11):  # 10 turns max
            if not continue_conversation:
                break

            try:
                # Play a sound to indicate the system is ready to listen
                await play_prompt_sound("start")

                # Print a visual prompt
                print("\n" + "=" * 50)
                print(f"🎤 TURN {turn}: SPEAK NOW - System is listening...")
                print("Listening for 5 seconds...")
                print("=" * 50 + "\n")

                # Capture audio
                audio_data = await audio_capture.capture_audio(5)  # Capture 5 seconds of audio

                # Process the captured audio
                data = {"audio_data": audio_data}
                result = await pipeline.process(data)

                # Play a sound to indicate the system has finished listening
                await play_prompt_sound("stop")

                # Print the result summary
                print("\n" + "=" * 50)
                print("🔊 SYSTEM RESPONSE SUMMARY")
                print("=" * 50)

                if result.get("transcription"):
                    print(f"Heard: \"{result.get('transcription')}\"")
                    print(f"Response: \"{result.get('response')}\"")

                    # Play the response
                    speech_file = result.get("speech_file")
                    if speech_file and os.path.exists(speech_file):
                        await play_audio_file(speech_file)
                else:
                    print("No speech detected or transcription failed")

                print("=" * 50 + "\n")

                # Print performance metrics
                performance_tracker.print_summary()

                # Ask if the user wants to continue
                print("Continue? (y/n): ", end="", flush=True)

                # Get user input synchronously to avoid issues
                user_input = input().strip().lower()

                # Check if user wants to continue
                if user_input != "y":
                    continue_conversation = False
                    print("\nEnding conversation as requested.")
                    break

            except asyncio.TimeoutError:
                print("\nTimeout waiting for input. Continuing...")

            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break

            except Exception as e:
                logger.error(f"Error in conversation turn {turn}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        # Print goodbye message
        print("\n" + "=" * 50)
        print("👋 CONVERSATION ENDED")
        print("=" * 50)
        print("Thank you for using the conversational audio pipeline!")
        print("=" * 50 + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        # Shutdown components
        await audio_capture.shutdown()
        await pipeline.shutdown()

        # Force garbage collection
        gc.collect()

        # Explicitly exit the program
        print("Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Suppress all warnings at the Python level
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Make sure components directory exists
    os.makedirs("components", exist_ok=True)

    print("=" * 50 + "\n")

    # Import time here to avoid circular import
    import time

    # Run the main function
    asyncio.run(run_conversational_pipeline())
