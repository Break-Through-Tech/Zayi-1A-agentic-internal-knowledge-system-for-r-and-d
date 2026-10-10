"""Gemini calls with exponential backoff (tenacity).

Usage:
    from config import load_config
    from gemini_client import get_llm, invoke_with_retry

    cfg = load_config()
    llm = get_llm(cfg)                       # ChatGoogleGenerativeAI, its own retries turned off
    chain = prompt | llm
    resp = invoke_with_retry(chain, {"question": q, "context": ctx}, cfg)

Only transient failures are retried: rate limits (429), server errors (500/502/503/504),
timeouts and dropped connections. Bad keys (401/403) and bad requests (400) fail
immediately, because retrying them can never succeed. When the attempts run out, the
original exception is re-raised.
"""
import logging
from functools import wraps

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from config import GeminiConfig, load_config

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# Fallback for libraries that don't expose a numeric status (matched by class name).
RETRYABLE_ERROR_NAMES = {
    "ResourceExhausted",  # 429
    "TooManyRequests",  # 429
    "InternalServerError",  # 500
    "ServiceUnavailable",  # 503
    "DeadlineExceeded",  # 504
    "GatewayTimeout",  # 504
}


def _status_code(exc):
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return int(value)
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return int(value) if isinstance(value, int) else None


def is_retryable(exc):
    """True for errors that may succeed on a later attempt."""
    seen = set()
    while exc is not None and id(exc) not in seen:  # also check the wrapped cause
        seen.add(id(exc))
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        if _status_code(exc) in RETRYABLE_STATUS_CODES:
            return True
        if type(exc).__name__ in RETRYABLE_ERROR_NAMES:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def retry_policy(config: GeminiConfig):
    """The tenacity decorator built from the config. Exposed so tests can inspect it."""
    return retry(
        retry=retry_if_exception(is_retryable),
        wait=wait_random_exponential(
            multiplier=config.wait_multiplier, min=config.wait_min, max=config.wait_max
        ),
        stop=stop_after_attempt(config.max_attempts),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


def with_gemini_retry(config: GeminiConfig = None):
    """Decorator: retry the wrapped function on transient Gemini errors."""
    config = config or load_config()

    def decorator(fn):
        retrying = retry_policy(config)(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return retrying(*args, **kwargs)

        return wrapper

    return decorator


def invoke_with_retry(runnable, inputs, config: GeminiConfig = None, **invoke_kwargs):
    """runnable.invoke(inputs) with backoff. Works with a chain, a prompt|llm, or an llm."""
    config = config or load_config()
    return retry_policy(config)(runnable.invoke)(inputs, **invoke_kwargs)


def get_llm(config: GeminiConfig = None):
    """Build the Gemini chat model from the config.

    max_retries=0 turns off langchain's built-in retries so they don't multiply with ours
    (otherwise 6 tenacity attempts x 6 internal retries = 36 calls per question).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI  # lazy: keep tests light

    config = config or load_config()
    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=config.api_key,
        temperature=config.temperature,
        timeout=config.timeout,
        max_retries=0,
    )
