import logging
import sys

def setup_logger():
    """Sets up a professional logger for the pipeline."""
    logger = logging.getLogger("verifier_repo")
    logger.setLevel(logging.INFO)
    
    # Format: 2026-02-02 12:00:00 - INFO - Message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

logger = setup_logger()