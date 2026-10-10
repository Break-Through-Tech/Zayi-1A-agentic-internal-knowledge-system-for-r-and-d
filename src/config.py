"""API key and Gemini settings, loaded in one place.

Usage:
    from config import load_config
    cfg = load_config()          # raises MissingAPIKeyError if no key is found

The API key is looked up in this order:
    1. GOOGLE_API_KEY environment variable (also read from a .env file at the repo root)
    2. Colab secret named GOOGLE_API_KEY (Colab "Secrets" tab)

Every other setting has a default and can be overridden with an environment variable
(see ENV_OVERRIDES), so nothing is hardcoded in notebooks.
"""
import os
from dataclasses import dataclass, field

try:  # python-dotenv is optional: without it, only real env vars / Colab secrets are used
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover
    find_dotenv = load_dotenv = None

API_KEY_VAR = "GOOGLE_API_KEY"


class MissingAPIKeyError(RuntimeError):
    """Raised when no Gemini API key can be found."""


@dataclass(frozen=True)
class GeminiConfig:
    # repr=False keeps the key out of logs, tracebacks and notebook output
    api_key: str = field(repr=False)
    model: str = "gemini-3.8-flash"  # same model Niv's notebook uses
    temperature: float = 0.0
    timeout: float = 60.0  # seconds per request
    # retry / exponential backoff settings (used by gemini_client)
    max_attempts: int = 6  # total tries, including the first
    wait_multiplier: float = 1.0  # scales the exponential growth: ~1s, 2s, 4s, ...
    wait_min: float = 1.0  # seconds
    wait_max: float = 60.0  # seconds, cap on a single wait


# config field -> (environment variable, type)
ENV_OVERRIDES = {
    "model": ("GEMINI_MODEL", str),
    "temperature": ("GEMINI_TEMPERATURE", float),
    "timeout": ("GEMINI_TIMEOUT", float),
    "max_attempts": ("GEMINI_MAX_ATTEMPTS", int),
    "wait_multiplier": ("GEMINI_WAIT_MULTIPLIER", float),
    "wait_min": ("GEMINI_WAIT_MIN", float),
    "wait_max": ("GEMINI_WAIT_MAX", float),
}


def _colab_secret(name):
    try:
        from google.colab import userdata  # only importable inside Colab

        return userdata.get(name)
    except Exception:  # not in Colab, secret missing, or notebook access not granted
        return None


def get_api_key():
    if load_dotenv is not None:
        load_dotenv(find_dotenv())
    key = (os.getenv(API_KEY_VAR) or _colab_secret(API_KEY_VAR) or "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"{API_KEY_VAR} not found. Copy .env.example to .env at the repo root and "
            f"set {API_KEY_VAR}=<your key>, or in Colab add it under Secrets. "
            "Get a key at https://aistudio.google.com/apikey"
        )
    return key


def load_config(**overrides):
    """Build a GeminiConfig from env vars, then apply explicit keyword overrides."""
    values = {}
    for name, (var, cast) in ENV_OVERRIDES.items():
        raw = os.getenv(var)
        if raw not in (None, ""):
            try:
                values[name] = cast(raw)
            except ValueError as e:
                raise ValueError(f"{var}={raw!r} is not a valid {cast.__name__}") from e
    values.update(overrides)
    if "api_key" not in values:
        values["api_key"] = get_api_key()
    cfg = GeminiConfig(**values)
    if cfg.max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    return cfg
