import os
import logging
import openai

logger = logging.getLogger(__name__)

class LLMUtilities:
    """
    A utility class for generating responses using the KISSKI LLM endpoint (OpenAI-compatible).
    """
    def __init__(self, model_name="meta-llama-3.1-70b-instruct", use_gpu=True):
        """
        Initialize the LLM utility with the specified model and GPU preference.
        Args:
            model_name (str): The KISSKI model name to load (e.g., "meta-llama-3.1-70b-instruct").
            use_gpu (bool): Whether GPU usage is requested. Actual GPU usage depends on KISSKI's service.
        """
        self.model_name = model_name
        self.use_gpu = use_gpu

        # Use the KISSKI-provided API key from environment
        openai.api_key = os.environ.get("KISSKI_API_KEY")
        if not openai.api_key:
            logger.error("Missing KISSKI_API_KEY environment variable.")
            raise EnvironmentError("Please set KISSKI_API_KEY for KISSKI LLM access.")

        # Point OpenAI client to the KISSKI Chat AI endpoint
        openai.api_base = "https://chat-ai.academiccloud.de/v1"

        logger.info(
            f"KISSKI LLM configured with model '{self.model_name}'. GPU usage = {self.use_gpu}."
        )

    def generate_response(self, prompt, max_new_tokens=150, num_return_sequences=1):
        """
        Generate a response from the KISSKI LLM service using the new openai>=1.0.0 Chat interface.
        Args:
            prompt (str): The input prompt for the model.
            max_new_tokens (int): Maximum tokens to generate in the reply.
            num_return_sequences (int): How many responses to return.
        Returns:
            str: The generated response text from the LLM.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                n=num_return_sequences,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during response generation via KISSKI LLM: {e}")
            return f"Sorry, I couldn't generate a response. Error: {e}"
