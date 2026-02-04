import os
import pandas as pd
from src.logger import logger

class DataLoader:
    """
    Handles file I/O operations for the verifier tools.
    Focuses on reading JSONL files and managing cached results.
    """

    @staticmethod
    def load_jsonl(file_path: str) -> pd.DataFrame:
        """
        Reads a JSONL file and returns a pandas DataFrame.
        """
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            return None
        
        try:
            df = pd.read_json(file_path, lines=True)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None

    @staticmethod
    def save_jsonl(df: pd.DataFrame, file_path: str):
        """
        Saves a DataFrame to JSONL format.
        """
        try:
            df.to_json(file_path, orient="records", lines=True)
            logger.info(f"Output successfully saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {str(e)}")

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        Checks if a file or directory exists to support 'Resume Safe' logic.
        """
        exists = os.path.exists(file_path)
        if exists:
            logger.info(f"Existing artifact found at {file_path}. Skipping step.")
        return exists