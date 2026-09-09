"""Logging utilities."""
import os
import logging

def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Configure logging to write to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
