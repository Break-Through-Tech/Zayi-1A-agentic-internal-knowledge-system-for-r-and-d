"""Unit tests for config.py and gemini_client.py. No network, no API key needed.

Run from the repo root:
    pytest -v tests/test_gemini_client.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config as cfgmod  # noqa: E402
import gemini_client  # noqa: E402
from config import GeminiConfig, MissingAPIKeyError, load_config  # noqa: E402

FAST = dict(api_key="test-key", wait_min=0, wait_max=0.01, wait_multiplier=0.001)


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class ResourceExhausted(Exception):  # matched by class name, no numeric code
    pass


class FlakyRunnable:
    """Fails with `errors` in order, then returns 'ok'."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def invoke(self, inputs, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


# ---------------------------------------------------------------- retry
@pytest.mark.parametrize("code", [429, 500, 503])
def test_retries_transient_errors_then_succeeds(code):
    r = FlakyRunnable([FakeAPIError(code), FakeAPIError(code)])
    assert gemini_client.invoke_with_retry(r, {}, GeminiConfig(**FAST)) == "ok"
    assert r.calls == 3


def test_retries_error_matched_by_class_name_and_timeouts():
    r = FlakyRunnable([ResourceExhausted(), TimeoutError()])
    assert gemini_client.invoke_with_retry(r, {}, GeminiConfig(**FAST)) == "ok"
    assert r.calls == 3


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_does_not_retry_permanent_errors(code):
    r = FlakyRunnable([FakeAPIError(code)])
    with pytest.raises(FakeAPIError):
        gemini_client.invoke_with_retry(r, {}, GeminiConfig(**FAST))
    assert r.calls == 1


def test_gives_up_after_max_attempts_and_reraises_original():
    r = FlakyRunnable([FakeAPIError(429)] * 10)
    with pytest.raises(FakeAPIError):
        gemini_client.invoke_with_retry(r, {}, GeminiConfig(max_attempts=3, **FAST))
    assert r.calls == 3


def test_decorator_form():
    r = FlakyRunnable([FakeAPIError(503)])
    wrapped = gemini_client.with_gemini_retry(GeminiConfig(**FAST))(r.invoke)
    assert wrapped({}) == "ok"
    assert r.calls == 2


# ---------------------------------------------------------------- config
@pytest.fixture
def clean_env(monkeypatch):
    for var in [cfgmod.API_KEY_VAR] + [v for v, _ in cfgmod.ENV_OVERRIDES.values()]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cfgmod, "load_dotenv", None)  # ignore any real .env
    monkeypatch.setattr(cfgmod, "_colab_secret", lambda name: None)
    return monkeypatch


def test_missing_key_raises_clear_error(clean_env):
    with pytest.raises(MissingAPIKeyError, match="GOOGLE_API_KEY"):
        load_config()


def test_key_and_overrides_from_env(clean_env):
    clean_env.setenv("GOOGLE_API_KEY", "  abc123  ")
    clean_env.setenv("GEMINI_MAX_ATTEMPTS", "4")
    clean_env.setenv("GEMINI_MODEL", "some-model")
    cfg = load_config()
    assert cfg.api_key == "abc123"
    assert cfg.max_attempts == 4 and cfg.model == "some-model"


def test_bad_env_value_is_reported(clean_env):
    clean_env.setenv("GOOGLE_API_KEY", "abc")
    clean_env.setenv("GEMINI_MAX_ATTEMPTS", "lots")
    with pytest.raises(ValueError, match="GEMINI_MAX_ATTEMPTS"):
        load_config()


def test_api_key_never_appears_in_repr():
    assert "secret-key" not in repr(GeminiConfig(api_key="secret-key"))
