import numpy as np
from typing import Dict, Any, Tuple

# Try to import logger, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("vad")

class VAD:
    def __init__(self, config=None, performance_tracker=None):
        self.name = "VAD"
        self.performance_tracker = performance_tracker
        self.energy_threshold = config.get("energy_threshold", 0.01) if config else 0.01
        self.window_size = config.get("window_size", 4) if config else 4
        self.advanced_mode = config.get("advanced_mode", True) if config else True

        logger.info(f"Initializing {self.name}")

    async def initialize(self):
        # Try to import webrtcvad for better VAD
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)
            self.use_webrtc = True
            logger.info("Using WebRTC VAD")
        except ImportError:
            self.use_webrtc = False
            logger.info("WebRTC VAD not available, using energy-based VAD")

        logger.info(f"{self.name} initialized successfully")
        return True

    async def process(self, audio_data) -> Tuple[bool, np.ndarray]:
        """Detect voice activity in audio data"""
        if self.performance_tracker:
            self.performance_tracker.start_component("VAD")

        try:
            # Check if audio data is empty
            if len(audio_data) == 0:
                has_speech = False
            else:
                if self.use_webrtc:
                    # Convert float audio to 16-bit PCM
                    audio_int16 = (audio_data * 32768).astype(np.int16)

                    # Process in 30ms frames (480 samples at 16kHz)
                    frame_size = 480
                    num_frames = len(audio_int16) // frame_size

                    # Check each frame for speech
                    speech_frames = 0
                    for i in range(num_frames):
                        frame = audio_int16[i * frame_size:(i + 1) * frame_size]
                        frame_bytes = frame.tobytes()
                        if self.vad.is_speech(frame_bytes, 16000):
                            speech_frames += 1

                    # If more than 30% of frames contain speech, consider it speech
                    has_speech = speech_frames / max(1, num_frames) > 0.3
                else:
                    # Simple energy-based VAD
                    energy = np.mean(np.abs(audio_data))
                    has_speech = energy > self.energy_threshold

            if self.performance_tracker:
                elapsed = self.performance_tracker.end_component("VAD")
                logger.info(f"VAD processing time: {elapsed:.3f}s")

            return has_speech, audio_data

        except Exception as e:
            logger.error(f"Error in VAD: {str(e)}")

            if self.performance_tracker:
                self.performance_tracker.end_component("VAD")

            return False, audio_data

    async def shutdown(self):
        logger.info(f"{self.name} shutdown successfully")
        return True
