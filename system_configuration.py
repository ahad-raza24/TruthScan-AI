"""
TruthScan System Configuration
=============================

This module contains all configuration settings for the TruthScan fake news detection system.
It provides a centralized location for managing system parameters, performance settings,
and API configurations.

Author: TruthScan Development Team
Version: 2.0
Last Updated: 2025
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ProcessingMode(Enum):
    """Enumeration of available processing modes for the system."""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class LogLevel(Enum):
    """Enumeration of available logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class KnowledgeGraphConfig:
    """Configuration settings for Knowledge Graph operations."""
    enabled: bool = True
    timeout_seconds: float = 6.0
    request_timeout_seconds: float = 3.0
    max_entities_to_check: int = 2
    max_relationships_per_pair: int = 2
    enable_caching: bool = True
    cache_expiry_hours: int = 24


@dataclass
class NewsRetrievalConfig:
    """Configuration settings for news article retrieval."""
    max_articles_per_query: int = 10
    enable_full_content_extraction: bool = False
    timeout_seconds: float = 10.0
    enable_deduplication: bool = True
    min_article_length: int = 50


@dataclass
class BiasAnalysisConfig:
    """Configuration settings for bias detection and analysis."""
    enable_advanced_nlp: bool = True
    enable_sentiment_analysis: bool = True
    enable_political_bias_detection: bool = True
    enable_source_credibility_check: bool = True
    bias_threshold: float = 0.6


@dataclass
class LLMDecisionConfig:
    """Configuration settings for Large Language Model decision making."""
    model_name: str = "deepseek/deepseek-r1-distill-qwen-32b:free"
    timeout_seconds: float = 30.0
    max_tokens: int = 1000
    temperature: float = 0.3
    enable_reasoning_traces: bool = True


@dataclass
class PerformanceConfig:
    """Configuration settings for system performance optimization."""
    enable_parallel_processing: bool = True
    max_concurrent_requests: int = 5
    enable_request_caching: bool = True
    cache_size_mb: int = 100
    enable_performance_monitoring: bool = True


@dataclass
class SecurityConfig:
    """Configuration settings for system security."""
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_input_sanitization: bool = True
    max_claim_length: int = 1000
    enable_api_key_rotation: bool = False


@dataclass
class LoggingConfig:
    """Configuration settings for system logging."""
    level: LogLevel = LogLevel.INFO
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    log_file_path: str = "truthscan_system.log"
    max_log_file_size_mb: int = 50
    backup_count: int = 5
    enable_performance_logging: bool = True
    enable_agent_communication_logging: bool = True


class SystemConfiguration:
    """
    Main configuration class that manages all system settings.
    
    This class provides a centralized way to access and modify system configuration
    settings. It supports environment variable overrides and different processing modes.
    """
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.BALANCED):
        """
        Initialize the system configuration.
        
        Args:
            mode: The processing mode to use (FAST, BALANCED, or ACCURATE)
        """
        self.mode = mode
        self._load_base_configuration()
        self._apply_mode_specific_settings()
        self._load_environment_overrides()
        self._validate_configuration()
    
    def _load_base_configuration(self) -> None:
        """Load the base configuration settings."""
        # Knowledge Graph Configuration
        self.knowledge_graph = KnowledgeGraphConfig()
        
        # News Retrieval Configuration
        self.news_retrieval = NewsRetrievalConfig()
        
        # Bias Analysis Configuration
        self.bias_analysis = BiasAnalysisConfig()
        
        # LLM Decision Configuration
        self.llm_decision = LLMDecisionConfig()
        
        # Performance Configuration
        self.performance = PerformanceConfig()
        
        # Security Configuration
        self.security = SecurityConfig()
        
        # Logging Configuration
        self.logging = LoggingConfig()
        
        # API Configuration
        self.api_keys = {
            'news_api': os.getenv('NEWS_API_KEY'),
            'knowledge_graph_api': os.getenv('KNOWLEDGE_GRAPH_API_KEY'),
            'openrouter_api': os.getenv('OPENROUTER_API_KEY'),
            'mediastack_api': os.getenv('MEDIASTACK_API_KEY'),
            'gnews_api': os.getenv('GNEWS_API_KEY')
        }
        
        # File Paths Configuration
        self.file_paths = {
            'lexicon_file': os.path.join(os.path.dirname(__file__), "lexicons", "lexicon.json"),
            'subjectivity_lexicon_file': os.path.join(os.path.dirname(__file__), "lexicons", "subjclueslen1-HLTEMNLP05.tff"),
            'model_directory': os.path.join(os.path.dirname(__file__), "models"),
            'results_directory': os.path.join(os.path.dirname(__file__), "results"),
            'logs_directory': os.path.join(os.path.dirname(__file__), "logs")
        }
        
        # Ensure directories exist
        for directory_path in [self.file_paths['results_directory'], self.file_paths['logs_directory']]:
            os.makedirs(directory_path, exist_ok=True)
    
    def _apply_mode_specific_settings(self) -> None:
        """Apply settings specific to the selected processing mode."""
        if self.mode == ProcessingMode.FAST:
            # Optimize for speed
            self.knowledge_graph.enabled = False
            self.knowledge_graph.timeout_seconds = 3.0
            self.news_retrieval.max_articles_per_query = 5
            self.news_retrieval.enable_full_content_extraction = False
            self.bias_analysis.enable_advanced_nlp = False
            self.llm_decision.timeout_seconds = 15.0
            self.performance.max_concurrent_requests = 10
            
        elif self.mode == ProcessingMode.BALANCED:
            # Balance between speed and accuracy
            self.knowledge_graph.enabled = True
            self.knowledge_graph.timeout_seconds = 6.0
            self.news_retrieval.max_articles_per_query = 8
            self.news_retrieval.enable_full_content_extraction = False
            self.bias_analysis.enable_advanced_nlp = True
            self.llm_decision.timeout_seconds = 25.0
            self.performance.max_concurrent_requests = 5
            
        elif self.mode == ProcessingMode.ACCURATE:
            # Optimize for accuracy
            self.knowledge_graph.enabled = True
            self.knowledge_graph.timeout_seconds = 12.0
            self.news_retrieval.max_articles_per_query = 15
            self.news_retrieval.enable_full_content_extraction = True
            self.bias_analysis.enable_advanced_nlp = True
            self.llm_decision.timeout_seconds = 45.0
            self.performance.max_concurrent_requests = 3
    
    def _load_environment_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        # Knowledge Graph overrides
        if os.getenv('KG_ENABLED'):
            self.knowledge_graph.enabled = os.getenv('KG_ENABLED').lower() == 'true'
        if os.getenv('KG_TIMEOUT'):
            self.knowledge_graph.timeout_seconds = float(os.getenv('KG_TIMEOUT'))
        
        # News Retrieval overrides
        if os.getenv('MAX_ARTICLES'):
            self.news_retrieval.max_articles_per_query = int(os.getenv('MAX_ARTICLES'))
        
        # LLM Decision overrides
        if os.getenv('LLM_MODEL'):
            self.llm_decision.model_name = os.getenv('LLM_MODEL')
        if os.getenv('LLM_TIMEOUT'):
            self.llm_decision.timeout_seconds = float(os.getenv('LLM_TIMEOUT'))
        
        # Logging overrides
        if os.getenv('LOG_LEVEL'):
            try:
                self.logging.level = LogLevel(os.getenv('LOG_LEVEL').upper())
            except ValueError:
                pass  # Keep default if invalid value
    
    def _validate_configuration(self) -> None:
        """Validate the configuration settings."""
        # Validate timeouts
        if self.knowledge_graph.timeout_seconds <= 0:
            raise ValueError("Knowledge Graph timeout must be positive")
        if self.llm_decision.timeout_seconds <= 0:
            raise ValueError("LLM decision timeout must be positive")
        
        # Validate article limits
        if self.news_retrieval.max_articles_per_query <= 0:
            raise ValueError("Maximum articles per query must be positive")
        
        # Validate file paths
        for path_name, path_value in self.file_paths.items():
            if path_name.endswith('_directory'):
                os.makedirs(path_value, exist_ok=True)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get an API key for a specific service.
        
        Args:
            service: The service name (e.g., 'news_api', 'knowledge_graph_api')
            
        Returns:
            The API key if available, None otherwise
        """
        return self.api_keys.get(service)
    
    def is_service_enabled(self, service: str) -> bool:
        """
        Check if a service is enabled and has the required API key.
        
        Args:
            service: The service name
            
        Returns:
            True if the service is enabled and configured, False otherwise
        """
        if service == 'knowledge_graph':
            return self.knowledge_graph.enabled and bool(self.get_api_key('knowledge_graph_api'))
        elif service == 'news_retrieval':
            return bool(self.get_api_key('news_api'))
        elif service == 'llm_decision':
            return bool(self.get_api_key('openrouter_api'))
        else:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'mode': self.mode.value,
            'knowledge_graph': self.knowledge_graph.__dict__,
            'news_retrieval': self.news_retrieval.__dict__,
            'bias_analysis': self.bias_analysis.__dict__,
            'llm_decision': self.llm_decision.__dict__,
            'performance': self.performance.__dict__,
            'security': self.security.__dict__,
            'logging': {k: v.value if isinstance(v, Enum) else v for k, v in self.logging.__dict__.items()},
            'file_paths': self.file_paths
        }
    
    def setup_logging(self) -> None:
        """Set up the logging configuration based on current settings."""
        log_level = getattr(logging, self.logging.level.value)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler if enabled
        if self.logging.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if self.logging.enable_file_logging:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                filename=os.path.join(self.file_paths['logs_directory'], self.logging.log_file_path),
                maxBytes=self.logging.max_log_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)


# Global configuration instance
# This can be imported and used throughout the application
config = SystemConfiguration(mode=ProcessingMode.BALANCED)

# Convenience functions for backward compatibility
def get_fast_config() -> SystemConfiguration:
    """Get a configuration optimized for speed."""
    return SystemConfiguration(mode=ProcessingMode.FAST)

def get_balanced_config() -> SystemConfiguration:
    """Get a balanced configuration."""
    return SystemConfiguration(mode=ProcessingMode.BALANCED)

def get_accurate_config() -> SystemConfiguration:
    """Get a configuration optimized for accuracy."""
    return SystemConfiguration(mode=ProcessingMode.ACCURATE)

# Legacy compatibility - these will be deprecated
ENABLE_KNOWLEDGE_GRAPH = config.knowledge_graph.enabled
KG_TIMEOUT = config.knowledge_graph.timeout_seconds
MAX_ARTICLES = config.news_retrieval.max_articles_per_query 