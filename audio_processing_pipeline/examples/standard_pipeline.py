"""
Standard pipeline example that demonstrates how to set up and run
the audio processing pipeline with file input/output.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from src.api.models import PipelineConfig
from src.components.audio_capture import AudioCapture, AudioCaptureConfig
from src.components.audio_preprocessing import AudioPreprocessing, AudioPreprocessingConfig
from src.components.speech_to_text import SpeechToText, SpeechToTextConfig
from src.components.text_processing import TextProcessing, TextProcessingConfig
from src.components.response_generator import ResponseGenerator, ResponseGeneratorConfig
from src.components.text_to_speech import TextToSpeech, TextToSpeechConfig
from src.config.default_config import get_pipeline_config, override_config
from src.core.pipeline import Pipeline
from src.utils.audio_exporter import AudioExporter
from src.utils.context import RequestContext
from src.utils.logger import logger, configure_logger


async def process_audio_file(input_file: str, output_dir: str = "output") -> Dict:
    """Process an audio file through the pipeline

    Args:
        input_file: Path to input audio file
        output_dir: Directory to save output files

    Returns:
        Dictionary with processing results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load audio file
    logger.info(f"Loading audio file: {input_file}")
    audio_data, sample_rate = sf.read(input_file)

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Convert to int16 format
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    # Get default pipeline configuration
    default_config = get_pipeline_config()

    # Override with custom settings for file processing
    custom_config = {
        "name": "FileProcessingPipeline",
        "components": [
            # Skip AudioCapture since we're using a file
            {
                "name": "AudioPreprocessing",
                "apply_noise_reduction": True,
                "apply_normalization": True
            },
            {
                "name": "SpeechToText",
                "chunk_size": len(audio_int16)  # Process the entire file at once
            },
            {
                "name": "TextProcessing",
                "agentic": True  # Enable context tracking
            },
            {
                "name": "ResponseGenerator",
                "temperature": 0.8  # Slightly more creative responses
            },
            {
                "name": "TextToSpeech",
                "use_streaming": False  # No need for streaming with file output
            }
        ]
    }

    # Merge configurations
    pipeline_config = override_config(default_config, custom_config)

    # Create pipeline configuration object
    config = PipelineConfig(**pipeline_config)

    # Create the pipeline
    pipeline = Pipeline(config)

    # Create and add components (skipping AudioCapture)
    preprocessing = AudioPreprocessing(AudioPreprocessingConfig(**pipeline_config["components"][0]))
    speech_to_text = SpeechToText(SpeechToTextConfig(**pipeline_config["components"][1]))
    text_processing = TextProcessing(TextProcessingConfig(**pipeline_config["components"][2]))
    response_generator = ResponseGenerator(ResponseGeneratorConfig(**pipeline_config["components"][3]))
    text_to_speech = TextToSpeech(TextToSpeechConfig(**pipeline_config["components"][4]))

    # Add components to pipeline
    pipeline.add_component(preprocessing)
    pipeline.add_component(speech_to_text)
    pipeline.add_component(text_processing)
    pipeline.add_component(response_generator)
    pipeline.add_component(text_to_speech)

    # Initialize the pipeline
    await pipeline.initialize()

    # Process the audio
    try:
        # Create a request context
        async with RequestContext(user_id="file_processor", session="file_session"):
            # Process through each component manually
            logger.info("Processing audio through pipeline...")

            # Audio preprocessing
            with pipeline.monitor.track(preprocessing.name):
                processed_audio = await preprocessing.process(audio_bytes)
                if processed_audio is None:
                    raise ValueError("Audio preprocessing failed")

            # Save processed audio
            processed_audio_path = os.path.join(output_dir, "processed_audio.wav")
            AudioExporter.save_wav(
                processed_audio,
                os.path.basename(processed_audio_path),
                framerate=sample_rate
            )
            logger.info(f"Saved processed audio to {processed_audio_path}")

            # Plot waveform
            waveform_path = os.path.join(output_dir, "waveform.png")
            AudioExporter.plot_audio(
                processed_audio,
                title="Processed Audio Waveform",
                save_path=waveform_path
            )

            # Speech to text
            with pipeline.monitor.track(speech_to_text.name):
                transcription = await speech_to_text.process(processed_audio)
                if transcription is None:
                    raise ValueError("Speech-to-text failed")

            logger.info(f"Transcription: {transcription}")

            # Text processing
            with pipeline.monitor.track(text_processing.name):
                processed_text = await text_processing.process(transcription)
                if processed_text is None:
                    raise ValueError("Text processing failed")

            # Save NLP results
            nlp_path = os.path.join(output_dir, "nlp_results.json")
            with open(nlp_path, 'w') as f:
                json.dump(processed_text, f, indent=2)
            logger.info(f"Saved NLP results to {nlp_path}")

            # Generate response
            with pipeline.monitor.track(response_generator.name):
                response = await response_generator.process(transcription)
                if response is None:
                    raise ValueError("Response generation failed")

            logger.info(f"Generated response: {response}")

            # Text to speech
            with pipeline.monitor.track(text_to_speech.name):
                response_audio = await text_to_speech.process(response)
                if response_audio is None:
                    raise ValueError("Text-to-speech failed")

            # Save response audio
            response_audio_path = os.path.join(output_dir, "response_audio.wav")
            AudioExporter.save_wav(
                response_audio,
                os.path.basename(response_audio_path),
                framerate=sample_rate
            )
            logger.info(f"Saved response audio to {response_audio_path}")

            # Get performance stats
            stats = pipeline.monitor.get_stats()

            # Save stats to file
            stats_path = os.path.join(output_dir, "pipeline_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

            # Create a summary of results
            results = {
                "input_file": input_file,
                "transcription": transcription,
                "entities": processed_text.get("entities", []),
                "response": response,
                "output_files": {
                    "processed_audio": processed_audio_path,
                    "waveform": waveform_path,
                    "nlp_results": nlp_path,
                    "response_audio": response_audio_path,
                    "stats": stats_path
                },
                "performance": {
                    "total_time": stats["uptime"],
                    "component_times": {
                        name: data["avg_time"]
                        for name, data in stats["components"].items()
                    }
                }
            }

            # Save summary
            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Processing complete. Results saved to {output_dir}")
            return results

    finally:
        # Shutdown the pipeline
        await pipeline.shutdown()


async def batch_process_directory(input_dir: str, output_dir: str = "output") -> List[Dict]:
    """Process all audio files in a directory

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save output files

    Returns:
        List of processing results for each file
    """
    # Get all audio files in the directory
    audio_files = []
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        audio_files.extend(list(Path(input_dir).glob(f"*{ext}")))

    logger.info(f"Found {len(audio_files)} audio files in {input_dir}")

    # Process each file
    results = []
    for audio_file in audio_files:
        # Create a subdirectory for each file's output
        file_output_dir = os.path.join(output_dir, audio_file.stem)

        logger.info(f"Processing {audio_file}...")
        try:
            result = await process_audio_file(str(audio_file), file_output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {str(e)}")

    # Create a summary of all results
    summary = {
        "processed_files": len(results),
        "total_files": len(audio_files),
        "results": results
    }

    # Save batch summary
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Batch processing complete. Summary saved to {summary_path}")
    return results


async def main():
    """Main function to run the standard pipeline example"""
    # Configure logger
    configure_logger(log_level="INFO", log_format="both")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Check if input file or directory is provided
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.isdir(input_path):
            await batch_process_directory(input_path)
        elif os.path.isfile(input_path):
            await process_audio_file(input_path)
        else:
            logger.error(f"Input path does not exist: {input_path}")
    else:
        # Use a default example file if available
        example_file = "examples/sample_audio.wav"
        if os.path.exists(example_file):
            await process_audio_file(example_file)
        else:
            logger.error(f"No input file provided and default example file not found: {example_file}")
            logger.info("Usage: python -m src.examples.standard_pipeline <input_file_or_directory>")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())