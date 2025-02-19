# config.py
class BaseConfig:
    """Base configuration class containing shared model settings"""

    MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"
    MAX_MEMORY = {"cuda:0": "3GB"}
    DEVICE_MAP = {
        "model.embed_tokens": "cuda:0",
        "model.layers": "cuda:0",
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
    }
    # System prompt
    SYSTEM_PROMPT = """You are an intelligent AI assistant. Please provide detailed and accurate responses.
    If the question is about:
    - Programming: Include code examples and explanations
    - Math: Show step-by-step solutions
    - General Knowledge: Provide comprehensive explanations with examples
    - Technical Topics: Break down complex concepts"""


class FastConfig(BaseConfig):
    """Fast mode configuration - optimized for quick responses"""

    MAX_NEW_TOKENS = 32
    TEMPERATURE = 0.8
    TOP_P = 0.95
    TOP_K = 50
    NUM_BEAMS = 1
    REPETITION_PENALTY = 1.1
    NO_REPEAT_NGRAM_SIZE = 2
    MAX_CONTEXT_LENGTH = 3
    MAX_INPUT_LENGTH = 192


class BalancedConfig(BaseConfig):
    """Balanced mode configuration - default settings"""

    MAX_NEW_TOKENS = 64
    TEMPERATURE = 0.7
    TOP_P = 0.92
    TOP_K = 40
    NUM_BEAMS = 2
    REPETITION_PENALTY = 1.2
    LENGTH_PENALTY = 1.0
    NO_REPEAT_NGRAM_SIZE = 3
    MAX_CONTEXT_LENGTH = 5
    MAX_INPUT_LENGTH = 384


class SmartConfig(BaseConfig):
    """Smart mode configuration - optimized for detailed responses"""

    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.5
    TOP_P = 0.85
    TOP_K = 30
    NUM_BEAMS = 4
    REPETITION_PENALTY = 1.3
    LENGTH_PENALTY = 1.2
    NO_REPEAT_NGRAM_SIZE = 4
    MAX_CONTEXT_LENGTH = 8
    MAX_INPUT_LENGTH = 512


def get_config(mode: str = "balanced"):
    """Get configuration based on specified mode"""
    configs = {"fast": FastConfig, "balanced": BalancedConfig, "smart": SmartConfig}
    return configs.get(mode.lower(), BalancedConfig)
