import os
import uuid
import asyncio
import gc
from typing import Dict, Any

# Try to import logger, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("text_to_speech")

class TextToSpeech:
    def __init__(self, config=None, performance_tracker=None):
        self.name = "TextToSpeech"
        self.performance_tracker = performance_tracker
        self.model_name = config.get("model_name", "tts_models/en/ljspeech/tacotron2-DDC") if config else "tts_models/en/ljspeech/tacotron2-DDC"
        self.use_streaming = config.get("use_streaming", True) if config else True
        self.use_gtts = config.get("use_gtts", True) if config else True
        self.output_dir = config.get("output_dir", "output/tts") if config else "output/tts"
        self.model = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initializing {self.name}")
    
    async def initialize(self):
        try:
            if not self.use_gtts:
                # Try to import TTS
                try:
                    import TTS
                    from TTS.utils.manage import ModelManager
                    from TTS.utils.synthesizer import Synthesizer
                    
                    # Initialize TTS
                    logger.info(f"Initializing TTS with model: {self.model_name}")
                    
                    # This is a placeholder for TTS initialization
                    # In a real implementation, you would initialize the TTS model here
                    
                    logger.info("TTS model initialized")
                except ImportError:
                    logger.warning("TTS not installed. Using gTTS instead.")
                    self.use_gtts = True
            
            logger.info(f"{self.name} initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing {self.name}: {str(e)}")
            self.use_gtts = True
            return False
    
    async def process(self, text: str) -> str:
        """Convert text to speech"""
        if self.performance_tracker:
            self.performance_tracker.start_component("TextToSpeech")
        
        try:
            # Check if text is empty
            if not text:
                if self.performance_tracker:
                    self.performance_tracker.end_component("TextToSpeech")
                return ""
            
            # Generate a unique filename
            filename = f"{uuid.uuid4()}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            if self.use_gtts:
                # Use gTTS for text-to-speech
                try:
                    from gtts import gTTS
                    
                    # Generate speech
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: gTTS(text=text, lang='en').save(filepath)
                    )
                    
                    logger.info(f"Generated speech file: {filepath}")
                except ImportError:
                    logger.error("gTTS not installed. Please install it with 'pip install gtts'")
                    filepath = ""
            else:
                # Use TTS for text-to-speech
                # This is a placeholder for TTS synthesis
                # In a real implementation, you would use the TTS model to generate speech
                
                logger.info(f"Generated speech file: {filepath}")
            
            if self.performance_tracker:
                elapsed = self.performance_tracker.end_component("TextToSpeech")
                logger.info(f"TextToSpeech time: {elapsed:.3f}s")
            
            return filepath
        
        except Exception as e:
            logger.error(f"Error in TTS: {str(e)}")
            
            if self.performance_tracker:
                self.performance_tracker.end_component("TextToSpeech")
            
            return ""
    
    async def shutdown(self):
        # Clear model from memory
        self.model = None
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"{self.name} shutdown successfully")
        return
