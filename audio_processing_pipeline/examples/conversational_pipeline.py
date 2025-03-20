import asyncio
import json
import os
from pathlib import Path

from src.api.models import PipelineConfig
from src.components.audio_capture import AudioCapture, AudioCaptureConfig
from src.components.audio_preprocessing import AudioPreprocessing, AudioPreprocessingConfig
from src.components.response_generator import ResponseGenerator, ResponseGeneratorConfig
from src.components.speech_to_text import SpeechToText, SpeechToTextConfig
from src.components.text_processing import TextProcessing, TextProcessingConfig
from src.components.text_to_speech import TextToSpeech, TextToSpeechConfig
from src.components.vad import VAD, VADConfig
from src.core.pipeline import ConversationalPipeline
from src.utils.context import RequestContext
from src.utils.logger import logger


async def run_conversational_pipeline():
    """Run the conversational pipeline with real microphone input"""

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Create the pipeline configuration
    pipeline_config = PipelineConfig(
        name="ConversationalAudioPipeline",
        log_level="INFO",
        log_format="console",
        components=[
            # Input components
            AudioCaptureConfig(
                name="AudioSource",
                rate=16000,
                frames_per_buffer=1024,
                export_audio=True,
                use_mock=False  # Use real microphone
            ),
            VADConfig(
                name="VoiceDetection",
                energy_threshold=0.01,
                window_size=4,
                advanced_mode=True
            ),
            AudioPreprocessingConfig(
                name="AudioPreprocessing",
                apply_gain=True,
                gain_factor=1.5,
                apply_noise_reduction=True
            ),
            SpeechToTextConfig(
                name="SpeechToText",
                model_name="facebook/wav2vec2-base-960h",
                use_mock=False
            ),
            TextProcessingConfig(
                name="TextProcessing",
                model_name="en_core_web_sm",
                agentic=False,
                use_mock=False
            ),
            # Response generation components
            ResponseGeneratorConfig(
                name="ResponseGenerator",
                model_name="facebook/opt-125m",
                use_cached_responses=True
            ),
            TextToSpeechConfig(
                name="TextToSpeech",
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                use_streaming=True,
                pre_generate_common_phrases=True
            )
        ],
        api_enabled=True,
        api_auth_required=False
    )

    # Create the conversational pipeline
    pipeline = ConversationalPipeline(pipeline_config)

    # Create and add all components
    audio_source = AudioCapture(pipeline_config.components[0])
    vad = VAD(pipeline_config.components[1])
    preprocessing = AudioPreprocessing(pipeline_config.components[2])
    speech_to_text = SpeechToText(pipeline_config.components[3])
    text_processing = TextProcessing(pipeline_config.components[4])
    response_generator = ResponseGenerator(pipeline_config.components[5])
    text_to_speech = TextToSpeech(pipeline_config.components[6])

    # Add components to pipeline
    pipeline.add_component(audio_source)
    pipeline.add_component(vad)
    pipeline.add_component(preprocessing)
    pipeline.add_component(speech_to_text)
    pipeline.add_component(text_processing)
    pipeline.add_component(response_generator)
    pipeline.add_component(text_to_speech)

    # Run the conversation with 10 turns max
    async with RequestContext(user_id="user", session="conversation"):
        logger.info("Starting conversational audio pipeline")
        logger.info("Speak into your microphone...")
        logger.info("Press Ctrl+C to stop the conversation")

        try:
            await pipeline.run_conversation(
                audio_source.audio_generator(),
                max_turns=10
            )
        except KeyboardInterrupt:
            logger.info("Conversation stopped by user")
        finally:
            # Get performance stats
            stats = pipeline.monitor.get_stats()
            # Log summary
            logger.info(f"Conversation completed: {stats['pipeline_name']} ran for {stats['uptime']:.2f} seconds")

            # Save stats to file
            with open(Path("output") / "pipeline_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

            logger.info("Stats saved to output/pipeline_stats.json")


if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(run_conversational_pipeline())