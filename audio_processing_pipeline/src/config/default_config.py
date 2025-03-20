"""
Default configuration for the audio processing pipeline.
This file contains the default settings for all components.
"""

from typing import Dict, List

# Pipeline configuration
PIPELINE_CONFIG = {
    "name": "AudioProcessingPipeline",
    "log_level": "INFO",
    "log_format": "console",
    "api_enabled": True,
    "api_auth_required": False,
    "api_key": None,
}

# Audio capture configuration
AUDIO_CAPTURE_CONFIG = {
    "name": "AudioCapture",
    "rate": 16000,         # samples per second
    "frames_per_buffer": 1024,  # frames per buffer
    "channels": 1,         # mono audio
    "format": 8,           # 16-bit audio (pyaudio.paInt16)
    "device_index": None,  # None uses default device
    "use_mock": False,     # Use mock audio instead of microphone
    "export_audio": False, # Whether to export captured audio
    "speech_texts": [
        "Welcome to the audio processing pipeline demo",
        "This is an example of speech recognition with transformers",
        "Natural language processing is done with spaCy",
        "The system can detect entities like names and locations",
        "Try saying something about New York or San Francisco"
    ]
}

# Voice activity detection configuration
VAD_CONFIG = {
    "name": "VoiceActivityDetection",
    "energy_threshold": 0.01,
    "window_size": 4,
    "speech_threshold": 0.5,
    "advanced_mode": True,
}

# Audio preprocessing configuration
AUDIO_PREPROCESSING_CONFIG = {
    "name": "AudioPreprocessing",
    "apply_gain": True,
    "gain_factor": 1.5,
    "apply_noise_reduction": True,
    "apply_normalization": False,
    "target_sample_rate": 16000,
}

# Speech-to-text configuration
SPEECH_TO_TEXT_CONFIG = {
    "name": "SpeechToText",
    "model_name": "facebook/wav2vec2-base-960h",
    "language": "en-US",
    "device": "cuda",  # Use GPU if available, otherwise falls back to CPU
    "chunk_size": 16000,  # Process in chunks of 1 second (16000 samples)
    "use_mock": False,
    "mock_texts": [
        "Hello world",
        "This is a test",
        "Audio processing pipeline",
        "How can I help you today?",
        "The weather is nice today"
    ]
}

# Text processing configuration
TEXT_PROCESSING_CONFIG = {
    "name": "TextProcessing",
    "model_name": "en_core_web_sm",
    "agentic": True,  # Use more complex context tracking and reasoning
    "use_mock": False,
    "mock_entities": {
        "PERSON": ["John", "Mary", "Steve Jobs"],
        "ORG": ["Apple", "Google", "Microsoft"],
        "GPE": ["New York", "San Francisco", "London"],
    }
}

# Response generator configuration
RESPONSE_GENERATOR_CONFIG = {
    "name": "ResponseGenerator",
    "model_name": "facebook/opt-125m",  # Lightweight model for testing
    "device": "cuda",  # Use GPU if available
    "max_length": 100,
    "temperature": 0.7,
    "use_cached_responses": True,  # Use cached responses for common inputs
    "response_cache_file": "cached_responses.json",
    "default_responses": {
        "greeting": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Greetings! How may I assist you?"
        ],
        "farewell": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Farewell! It was nice talking to you!"
        ],
        "fallback": [
            "I'm not sure I understand. Could you rephrase that?",
            "I didn't quite catch that. Can you say it differently?",
            "I'm having trouble understanding. Could you try again?"
        ]
    }
}

# Text-to-speech configuration
TEXT_TO_SPEECH_CONFIG = {
    "name": "TextToSpeech",
    "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
    "device": "cuda",  # Use GPU if available
    "use_streaming": True,
    "sample_rate": 16000,
    "pre_generate_common_phrases": True,
    "common_phrases_file": "common_phrases.json",
    "common_phrases": [
        "Hello! How can I help you today?",
        "I'm sorry, I didn't understand that.",
        "Could you please repeat that?",
        "Thank you for your question.",
        "Is there anything else you'd like to know?"
    ]
}

# API server configuration
API_SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 1,
    "cors_origins": ["*"],  # In production, restrict this to specific origins
    "api_prefix": "/api/v1",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
}

# Security configuration
SECURITY_CONFIG = {
    "enable_rate_limiting": True,
    "rate_limit": 100,  # requests per minute
    "enable_input_validation": True,
    "max_input_length": 1000,  # characters
    "enable_output_sanitization": True,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        },
    }
}

# Function to get the default configuration for a specific component
def get_component_config(component_name: str) -> Dict:
    """Get the default configuration for a specific component"""
    config_map = {
        "AudioCapture": AUDIO_CAPTURE_CONFIG,
        "VoiceActivityDetection": VAD_CONFIG,
        "AudioPreprocessing": AUDIO_PREPROCESSING_CONFIG,
        "SpeechToText": SPEECH_TO_TEXT_CONFIG,
        "TextProcessing": TEXT_PROCESSING_CONFIG,
        "ResponseGenerator": RESPONSE_GENERATOR_CONFIG,
        "TextToSpeech": TEXT_TO_SPEECH_CONFIG,
    }

    return config_map.get(component_name, {})

# Function to get the full pipeline configuration
def get_pipeline_config() -> Dict:
    """Get the full pipeline configuration"""
    return {
        **PIPELINE_CONFIG,
        "components": [
            AUDIO_CAPTURE_CONFIG,
            VAD_CONFIG,
            AUDIO_PREPROCESSING_CONFIG,
            SPEECH_TO_TEXT_CONFIG,
            TEXT_PROCESSING_CONFIG,
            RESPONSE_GENERATOR_CONFIG,
            TEXT_TO_SPEECH_CONFIG,
        ]
    }

# Function to override default configuration with user-provided values
def override_config(default_config: Dict, user_config: Dict) -> Dict:
    """Override default configuration with user-provided values"""
    result = default_config.copy()

    for key, value in user_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively update nested dictionaries
            result[key] = override_config(result[key], value)
        else:
            # Update value
            result[key] = value

    return result