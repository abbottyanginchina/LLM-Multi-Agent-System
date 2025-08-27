import os
from dotenv import load_dotenv
from typing import List, Optional, Dict
from openai import OpenAI
from backends.llm_registry import LLMRegistry
from backends.message import Message

load_dotenv()
MINE_API_KEYS = os.getenv("MINE_API_KEYS")
MINE_BASE_URL = os.getenv("MINE_BASE_URL")

@LLMRegistry.register('openAIChat')
class openAIChat:
    """
    API client class for managing LLM calls
    """
    def __init__(self, model_name: str):
        self.client = OpenAI(
            api_key=MINE_API_KEYS, 
            base_url=MINE_BASE_URL
        )
        self.model = model_name

    def generate(self, messages: List[Dict]) -> str:
        """
        Generate a response from the model based on user input and optional system prompt.
        """           
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False
        )
        
        return response.choices[0].message.content