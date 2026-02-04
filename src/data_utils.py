import os
import pandas as pd
from src.logger import logger

class DataHandler:
    """
    Handles file I/O for the verifier pipeline, ensuring 
    JSONL files are read and saved correctly.
    """

    @staticmethod
    def load_jsonl(file_path: str):
        """Loads a JSONL file into a DataFrame."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            df = pd.read_json(file_path, lines=True)
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load JSONL from {file_path}: {e}")
            return None

    @staticmethod
    def save_jsonl(df: pd.DataFrame, file_path: str):
        """Saves a DataFrame to a JSONL file."""
        try:
            df.to_json(file_path, orient="records", lines=True)
            # Guideline #5: Replaces print_artifact with structured logging
            abs_path = os.path.abspath(file_path)
            logger.info(f"Data saved to: {file_path}")
            logger.debug(f"Full path: {abs_path}")
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")

    @staticmethod
    def check_exists(file_path: str):
        """Checks if a file exists to support 'Resume Safe' operations."""
        exists = os.path.exists(file_path)
        if exists:
            logger.info(f"Cache found for: {file_path}. Skipping generation.")
        return exists