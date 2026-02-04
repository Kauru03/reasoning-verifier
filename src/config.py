import os
from dotenv import load_dotenv

# Load the .env file if it exists
load_dotenv()

class Config:
    """Centralized configuration management."""
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_VERSION = os.getenv("AZURE_VERSION", "2024-02-15-preview")
    
    # Model Names
    LARGER_MODEL_NAME = os.getenv("LARGER_MODEL_NAME", "gpt-4.1")
    SMALLER_MODEL_BASE = os.getenv("SMALLER_MODEL_BASE", "Qwen/Qwen2.5-3B-Instruct")