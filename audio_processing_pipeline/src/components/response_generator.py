import gc
import random
from datetime import datetime
from typing import Dict, Any, Union, List

# Try to import logger, fallback to standard logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("response_generator")

class ResponseGenerator:
    def __init__(self, config=None, performance_tracker=None):
        self.name = "ResponseGenerator"
        self.performance_tracker = performance_tracker
        self.model_name = config.get("model_name", "gpt2") if config else "gpt2"
        self.use_transformer = config.get("use_transformer", True) if config else True
        self.model = None
        self.tokenizer = None
        self.generator = None

        # Conversation history
        self.conversation_history = []

        # Knowledge base for specific topics
        self.knowledge_base = {
            "weather": "I don't have access to real-time weather data, but I can help you find a weather service if you'd like.",
            "time": f"The current time is {datetime.now().strftime('%H:%M')}.",
            "name": "My name is AI Assistant. I'm here to help you with various tasks.",
            "joke": "Why don't scientists trust atoms? Because they make up everything!",
            "help": "I'm an AI assistant. You can ask me questions or have a conversation with me."
        }

        logger.info(f"Initializing {self.name}")

    async def initialize(self):
        try:
            # Try to import transformers for better response generation
            if self.use_transformer:
                try:
                    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

                    # Load model for text generation
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

                    logger.info(f"Using transformer model {self.model_name} for response generation")
                except (ImportError, Exception) as e:
                    logger.warning(f"Transformer model not available: {str(e)}. Using rule-based responses.")
                    self.use_transformer = False

            logger.info(f"{self.name} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing {self.name}: {str(e)}")
            self.use_transformer = False
            return False

    def _generate_response_for_intent(self, intent: str, text: str, keywords: List[str]) -> str:
        """Generate a response based on intent and keywords"""
        if intent == "greeting":
            return "Hello! How can I help you today?"

        if intent == "farewell":
            return "Goodbye! Have a great day!"

        if intent == "help_request":
            return "I'm here to assist you. What would you like help with?"

        if intent == "question":
            # Check for specific question topics
            if any(word in keywords for word in ["weather", "temperature", "forecast"]):
                return self.knowledge_base["weather"]

            if any(word in keywords for word in ["time", "clock", "hour"]):
                return self.knowledge_base["time"]

            if any(word in keywords for word in ["name", "called", "who"]):
                return self.knowledge_base["name"]

            if any(word in keywords for word in ["joke", "funny"]):
                return self.knowledge_base["joke"]

            # General question response
            return f"That's an interesting question about {', '.join(keywords[:2]) if keywords else 'that topic'}. Let me think about it."

        # Default response for other intents
        return f"I understand you're talking about {', '.join(keywords[:2]) if keywords else 'something'}. How can I help with that?"

    async def process(self, data: Union[str, Dict[str, Any]]) -> str:
        """Generate a response based on input data"""
        if self.performance_tracker:
            self.performance_tracker.start_component("ResponseGenerator")

        try:
            # Extract text and processed data from input
            input_text = ""
            processed_data = {}

            if isinstance(data, str):
                input_text = data
            elif isinstance(data, dict):
                input_text = data.get("processed_text", "")
                if not input_text and "transcription" in data:
                    input_text = data["transcription"]

                # Extract processed data
                processed_data = {
                    "intent": data.get("intent", "general_query"),
                    "keywords": data.get("keywords", []),
                    "sentiment": data.get("sentiment", {"label": "neutral", "score": 0.5}),
                    "entities": data.get("entities", [])
                }

            # Check if input text is empty
            if not input_text:
                response = "I didn't catch that. Could you please repeat?"
            else:
                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": input_text})

                if self.use_transformer and self.generator:
                    try:
                        # Generate response using transformer model
                        prompt = f"User: {input_text}\nAssistant:"

                        # Generate response
                        generated_text = self.generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

                        # Extract the assistant's response
                        if "Assistant:" in generated_text:
                            response = generated_text.split("Assistant:")[-1].strip()
                        else:
                            response = generated_text.replace(prompt, "").strip()

                        # Fallback if response is too short or empty
                        if len(response) < 10:
                            response = self._generate_response_for_intent(
                                processed_data.get("intent", "general_query"),
                                input_text,
                                processed_data.get("keywords", [])
                            )
                    except Exception as e:
                        logger.error(f"Error generating response with transformer: {str(e)}")
                        response = self._generate_response_for_intent(
                            processed_data.get("intent", "general_query"),
                            input_text,
                            processed_data.get("keywords", [])
                        )
                else:
                    # Generate response using rule-based approach
                    response = self._generate_response_for_intent(
                        processed_data.get("intent", "general_query"),
                        input_text,
                        processed_data.get("keywords", [])
                    )

                # Add to conversation history
                self.conversation_history.append({"role": "assistant", "content": response})

                # Limit conversation history
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]

            if self.performance_tracker:
                elapsed = self.performance_tracker.end_component("ResponseGenerator")
                logger.info(f"ResponseGenerator time: {elapsed:.3f}s")

            # Log the generated response
            print("\n" + "=" * 50)
            print("💬 GENERATED RESPONSE")
            print("=" * 50)
            print(f"\"{response}\"")
            print("=" * 50 + "\n")

            logger.info(f"💬 RESPONSE: \"{response}\"")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

            if self.performance_tracker:
                self.performance_tracker.end_component("ResponseGenerator")

            return "I'm sorry, I encountered an error. Please try again."

    async def shutdown(self):
        # Clear model from memory
        self.model = None
        self.tokenizer = None
        self.generator = None

        # Force garbage collection
        gc.collect()

        logger.info(f"{self.name} shutdown successfully")
        return True
