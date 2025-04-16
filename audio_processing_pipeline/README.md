# Audio Processing Pipeline

A modular, production-ready audio streaming and processing pipeline for real-time speech recognition, natural language processing, and text-to-speech synthesis.

## Features

- Real-time audio capture from microphone
- Voice activity detection
- Speech recognition using Transformer models
- Natural language processing with spaCy
- Response generation with language models
- Text-to-speech synthesis
- REST API for remote interaction
- Modular architecture for easy extension

## Architecture


The pipeline uses a modular, event-driven architecture where each component:
- Operates independently
- Handles a specific task in the audio processing chain
- Communicates through well-defined interfaces
- Can be replaced or modified without affecting other components

### Component Flow

1. **Audio Capture** → Records raw audio from microphone
2. **Voice Activity Detection** → Detects speech in audio stream
3. **Audio Preprocessing** → Enhances audio quality
4. **Speech-to-Text** → Converts speech to text (GPU accelerated)
5. **Text Processing** → Analyzes text using NLP
6. **Response Generator** → Creates appropriate responses (GPU accelerated)
7. **Text-to-Speech** → Converts text responses to audio (GPU accelerated)

# Test suite and testing methodology on local system with less resources

# for general test of the flow 
**Note :Tested on python version 3.12.0**
1. After the necessary installations specified in requirements_lightweight.txt for running on local env, run the file examples/conversational_pipeline.py

# for testing the APIs
1. Run the server.py in api/server.py
2. run the automated tests as here: pytest -xvs tests/test_api.py


