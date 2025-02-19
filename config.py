# config.py


class BaseConfig:
    """
    Base configuration class containing shared model settings
    """

    # Model configuration - shared across all modes
    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
    MAX_MEMORY = {"cuda:0": "3GB"}
    DEVICE_MAP = {
        "model.embed_tokens": "cuda:0",
        "model.layers": "cuda:0",
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
    }


class FastConfig(BaseConfig):
    """
    Fast mode configuration
    Optimized for speed and responsiveness
    """

    # Generation parameters
    MAX_NEW_TOKENS = 32
    TEMPERATURE = 0.8
    TOP_P = 0.95
    TOP_K = 50
    NUM_BEAMS = 1
    REPETITION_PENALTY = 1.1
    LENGTH_PENALTY = 0.8
    NO_REPEAT_NGRAM_SIZE = 2

    # Context window settings
    MAX_CONTEXT_LENGTH = 3
    MAX_INPUT_LENGTH = 128

    # System prompt
    SYSTEM_PROMPT = """You are a fast and efficient AI assistant. Provide concise and direct responses.
    Focus on key points and keep explanations brief:
    - Programming: Show minimal working examples
    - Math: Quick solutions
    - Knowledge: Core concepts only
    - Technical: Key points only
    """


class BalancedConfig(BaseConfig):
    """
    Balanced mode configuration
    Default settings for general use
    """

    # Generation parameters
    MAX_NEW_TOKENS = 64
    TEMPERATURE = 0.7
    TOP_P = 0.92
    TOP_K = 40
    NUM_BEAMS = 2
    REPETITION_PENALTY = 1.2
    LENGTH_PENALTY = 1.0
    NO_REPEAT_NGRAM_SIZE = 3

    # Context window settings
    MAX_CONTEXT_LENGTH = 5
    MAX_INPUT_LENGTH = 256

    # System prompt
    SYSTEM_PROMPT = """You are an intelligent AI assistant. Please provide detailed and accurate responses.
    If the question is about:
    - Programming: Include code examples and explanations
    - Math: Show step-by-step solutions
    - General Knowledge: Provide comprehensive explanations with examples
    - Technical Topics: Break down complex concepts
    """


class SmartConfig(BaseConfig):
    """
    Smart mode configuration
    Optimized for detailed and thorough responses
    """

    # Generation parameters
    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.5
    TOP_P = 0.85
    TOP_K = 30
    NUM_BEAMS = 4
    REPETITION_PENALTY = 1.3
    LENGTH_PENALTY = 1.2
    NO_REPEAT_NGRAM_SIZE = 4

    # Context window settings
    MAX_CONTEXT_LENGTH = 8
    MAX_INPUT_LENGTH = 512

    # System prompt
    SYSTEM_PROMPT = """You are an advanced AI assistant focused on comprehensive and detailed responses.
    For each topic provide thorough analysis:
    - Programming: Detailed implementations with best practices, error handling, and optimization
    - Math: Complete step-by-step solutions with explanations and alternative approaches
    - Knowledge: In-depth explanations with context, examples, and practical applications
    - Technical Topics: Comprehensive breakdown with theory, implementation, and considerations
    Consider edge cases and include relevant examples in your responses.
    """


def get_config(mode: str = "balanced"):
    """
    Get configuration based on specified mode

    Args:
        mode (str): Configuration mode ("fast", "balanced", or "smart")

    Returns:
        Configuration class for specified mode
    """
    configs = {"fast": FastConfig, "balanced": BalancedConfig, "smart": SmartConfig}
    return configs.get(mode.lower(), BalancedConfig)


# Example usage:
"""
from config import get_config

# Get configuration for specific mode
config = get_config("fast")  # or "balanced" or "smart"

# Use in DeepSeekAssistant
assistant = DeepSeekAssistant(config=config)
"""
