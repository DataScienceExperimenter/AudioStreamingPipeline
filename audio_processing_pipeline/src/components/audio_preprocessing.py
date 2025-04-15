import numpy as np
from typing import Dict, Any

# Try to import logger, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("audio_preprocessing")

class AudioPreprocessing:
    def __init__(self, config=None, performance_tracker=None):
        self.name = "AudioPreprocessing"
        self.performance_tracker = performance_tracker
        self.apply_gain = config.get("apply_gain", True) if config else True
        self.gain_factor = config.get("gain_factor", 1.5) if config else 1.5
        self.apply_noise_reduction = config.get("apply_noise_reduction", True) if config else True

        logger.info(f"Initializing {self.name}")

    async def initialize(self):
        # Try to import librosa for better audio processing
        try:
            import librosa
            self.use_librosa = True
            logger.info("Using librosa for audio preprocessing")
        except ImportError:
            self.use_librosa = False
            logger.info("Librosa not available, using basic audio preprocessing")

        logger.info(f"{self.name} initialized successfully")
        return True

    async def process(self, audio_data):
        """Preprocess audio data"""
        if self.performance_tracker:
            self.performance_tracker.start_component("AudioPreprocessing")

        try:
            # Check if audio data is empty
            if len(audio_data) == 0:
                if self.performance_tracker:
                    self.performance_tracker.end_component("AudioPreprocessing")
                return audio_data

            if self.use_librosa:
                import librosa
                import librosa.effects

                # Convert to float32 if needed
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Normalize audio
                audio_data = librosa.util.normalize(audio_data)

                # Apply noise reduction (spectral gating)
                if self.apply_noise_reduction:
                    try:
                        import noisereduce as nr
                        audio_data = nr.reduce_noise(y=audio_data, sr=16000)
                    except ImportError:
                        # Simple noise gate
                        noise_threshold = 0.01
                        audio_data[np.abs(audio_data) < noise_threshold] = 0

                # Apply gain
                if self.apply_gain:
                    audio_data = audio_data * self.gain_factor
                    # Clip to prevent distortion
                    audio_data = np.clip(audio_data, -1.0, 1.0)
            else:
                # Apply gain if enabled
                if self.apply_gain and len(audio_data) > 0:
                    audio_data = audio_data * self.gain_factor
                    # Clip to prevent distortion
                    audio_data = np.clip(audio_data, -1.0, 1.0)

                # Apply simple noise reduction if enabled
                if self.apply_noise_reduction and len(audio_data) > 0:
                    # Simple noise reduction by removing low amplitude signals
                    noise_threshold = 0.01
                    audio_data[np.abs(audio_data) < noise_threshold] = 0

            if self.performance_tracker:
                elapsed = self.performance_tracker.end_component("AudioPreprocessing")
                logger.info(f"AudioPreprocessing time: {elapsed:.3f}s")

            return audio_data

        except Exception as e:
            logger.error(f"Error in AudioPreprocessing: {str(e)}")

            if self.performance_tracker:
                self.performance_tracker.end_component("AudioPreprocessing")

            return audio_data

    async def shutdown(self):
        logger.info(f"{self.name} shutdown successfully")
        return True
