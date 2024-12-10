import os
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class LLMUtilities:
    def __init__(self, model_name="meta-llama/Llama-2-7b", use_gpu=False):
        """
        Initialize the utility with the specified model and device configuration.

        Args:
            model_name (str): The Hugging Face model name to load.
            use_gpu (bool): Whether to use GPU for inference.
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            logger.error("HF_TOKEN is not set in the environment variables. Please set it to access the Hugging Face model.")
            raise EnvironmentError("Missing Hugging Face token.")
        self.pipeline = self._load_model()

    def _load_model(self):
        """
        Load the specified model and tokenizer, optimized for GPU if enabled.

        Returns:
            pipeline: The Hugging Face pipeline for text generation.
        """
        logger.info(f"Loading model '{self.model_name}' with {'GPU' if self.use_gpu else 'CPU'} inference.")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_auth_token=self.token
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.token
            )
            return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if self.use_gpu else -1)
        except Exception as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise e

    def generate_response(self, prompt, max_length=200, num_return_sequences=1):
        """
        Generate a response based on the provided prompt.

        Args:
            prompt (str): The input prompt for the model.
            max_length (int): Maximum length of the generated response.
            num_return_sequences (int): Number of response sequences to generate.

        Returns:
            str: The generated response.
        """
        try:
            response = self.pipeline(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
            return response[0]["generated_text"]
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return "Sorry, I couldn't generate a response."
