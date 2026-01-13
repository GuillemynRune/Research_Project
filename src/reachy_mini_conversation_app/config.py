import os
import logging

from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

# Locate .env file (search upward from current working directory)
dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    # Load .env and override environment variables
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Configuration loaded from {dotenv_path}")
else:
    logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Model backend selection
    USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "true").lower() in ("true", "1", "yes")
    
    # OpenAI settings (only used if USE_LOCAL_MODELS=false)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    
    # Local model settings (used if USE_LOCAL_MODELS=true)
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")  # tiny, base, small, medium, large
    GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-3-4b-it")  # Gemma 3 VLM
    
    # Optional: HuggingFace settings for local models
    HF_HOME = os.getenv("HF_HOME", "./cache")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Required for Gemma 3

    # TTS settings
    TTS_ENGINE = os.getenv("TTS_ENGINE", "none")  # none, piper, coqui, bark
    TTS_MODEL = os.getenv("TTS_MODEL", "")
    
    logger.debug(f"USE_LOCAL_MODELS: {USE_LOCAL_MODELS}")
    logger.debug(f"Whisper Model: {WHISPER_MODEL}, Gemma 3 VLM: {GEMMA_MODEL}")
    logger.debug(f"HF_HOME: {HF_HOME}")

    REACHY_MINI_CUSTOM_PROFILE = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env."""
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        import os as _os

        if profile:
            _os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            _os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass