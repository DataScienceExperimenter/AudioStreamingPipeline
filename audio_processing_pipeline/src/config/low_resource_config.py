"""
Low-resource configuration for systems without GPU and limited RAM.
"""

from src.config.default_config import get_pipeline_config, override_config

def get_low_resource_config():
    """Get configuration optimized for low-resource systems"""
    default_config = get_pipeline_config()

    # Override with lightweight settings
    low_resource_overrides = {
        "components": [
            # AudioCapture - reduce buffer size
            {
                "name": "AudioCapture",
                "frames_per_buffer": 512,
                "export_audio": False
            },
            # VAD - use simpler detection
            {
                "name": "VoiceActivityDetection",
                "advanced_mode": False
            },
            # AudioPreprocessing - minimal processing
            {
                "name": "AudioPreprocessing",
                "apply_noise_reduction": False,
                "apply_normalization": True
            },
            # SpeechToText - use tiny model
            {
                "name": "SpeechToText",
                "model_name": "facebook/wav2vec2-base-10k-voxpopuli",
                "device": "cpu",
                "chunk_size": 8000  # Process smaller chunks
            },
            # TextProcessing - use tiny model
            {
                "name": "TextProcessing",
                "model_name": "en_core_web_sm",
                "agentic": False  # Disable complex reasoning
            },
            # ResponseGenerator - use tiny model or rule-based
            {
                "name": "ResponseGenerator",
                "model_name": "distilgpt2",  # Much smaller model
                "device": "cpu",
                "use_cached_responses": True
            },
            # TextToSpeech - use gTTS instead of ML models
            {
                "name": "TextToSpeech",
                "use_gtts": True,  # Use Google TTS instead of ML model
                "device": "cpu"
            }
        ]
    }

    return override_config(default_config, low_resource_overrides)