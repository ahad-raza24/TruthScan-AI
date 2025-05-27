"""
Fake News Detection Agent System - Base Agent Architecture
"""

import os
import torch
import numpy as np
import pandas as pd
import pickle
import json
import re
import requests
import logging
from abc import ABC, abstractmethod
from scipy import sparse
from typing import Dict, List, Union, Tuple, Optional, Any
from tqdm import tqdm
import time
from datetime import datetime
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "fakenews_agent.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Abstract base class for all agents in the fake news detection system.
    
    Each concrete agent must implement the process() method to define
    its specific behavior (e.g., text analysis, source verification, etc.).
    """
    
    def __init__(self, name: str):
        """Initialize agent with a unique name."""
        self.name = name
        self.logger = logging.getLogger(f"{name}_agent")
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and return results.
        Must be implemented by concrete agent classes.
        """
        pass
    
    def log(self, message: str, level: str = "info"):
        """Log a message with the specified level (info, warning, error, debug)."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)