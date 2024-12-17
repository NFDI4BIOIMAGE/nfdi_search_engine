import os
import logging
import openai

logger = logging.getLogger(__name__)

class LLMUtilities:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-70B-Instruct"):
        """
        Initialize the utility to communicate with KISSKI/ScaDS.AI LLM service.
        Args:
            model_name (str): The LLM model name to use.
        """
        self.model_name = model_name
        self.api_key = os.getenv("SCADSAI_API_KEY")  # The environment variable must contain the HPC token
        if not self.api_key:
            logger.error("SCADSAI_API_KEY is not set. Please set it to access the HPC LLM endpoint.")
            raise EnvironmentError("Missing SCADSAI_API_KEY environment variable.")

        # Configure the openai library to use the KISSKI HPC endpoint
        openai.api_key = self.api_key
        openai.api_base = "https://llm.scads.ai/v1"

    def generate_response(self, prompt, max_tokens=512):
        """
        Generate a response based on the provided prompt by sending a request
        to the HPC KISSKI LLM endpoint.
        Args:
            prompt (str): The input prompt for the HPC LLM.
            max_tokens (int): The maximum number of tokens to generate.
        Returns:
            str: The generated response from the LLM.
        """
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return f"Sorry, I couldn't generate a response. Error: {e}"
