"""LLM-based shot classification fallback (tier 2 of the classification
chain: ML -> LLM -> rule-based).

Used only when classify_shot_ml() didn't produce a confident result (no
trained model, or low confidence). Requires GEMINI_API_KEY (free tier -
see .env.example); with no key, or LLM_FALLBACK_ENABLED=false, this tier
is a silent no-op and the pipeline falls straight through to the
rule-based classifier, exactly as it did before this tier existed.

Real trade-off worth naming plainly: using this sends the impact photo
to Google's servers.
"""

import hashlib
import json
import os

import cv2
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite")
LLM_FALLBACK_ENABLED = os.environ.get("LLM_FALLBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
LLM_MAX_CALLS_PER_SESSION = int(os.environ.get("LLM_MAX_CALLS_PER_SESSION", "400"))

_CACHE_MAX_SIZE = 100
_cache = {}
_cache_order = []
_session_call_count = 0
_session_disabled = False
_warned_missing_key = False


class LLMProvider:
    """Common interface every LLM provider implements."""

    def classify(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """The only working provider - free tier, no credit card required."""

    def __init__(self, api_key: str, model: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def classify(self, image_bytes: bytes, prompt: str) -> str:
        from google.genai import types
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return response.text


class AnthropicProvider(LLMProvider):
    """TODO: implement against the Anthropic Messages API if a paid key is ever added."""

    def classify(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError("AnthropicProvider is a stub - not implemented yet.")


class OpenAIProvider(LLMProvider):
    """TODO: implement against the OpenAI API if a paid key is ever added."""

    def classify(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError("OpenAIProvider is a stub - not implemented yet.")


_provider_cache = None
_provider_load_attempted = False


def _get_provider():
    """Lazily builds and caches the configured provider. Never raises -
    returns None (logging once) for any config/init problem."""
    global _provider_cache, _provider_load_attempted, _warned_missing_key
    if _provider_load_attempted:
        return _provider_cache
    _provider_load_attempted = True

    if LLM_PROVIDER != "gemini":
        print(f"LLM fallback: provider '{LLM_PROVIDER}' has no implementation yet - skipping.")
        return None

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        if not _warned_missing_key:
            print("LLM fallback: GEMINI_API_KEY not set - skipping (see .env.example to add one).")
            _warned_missing_key = True
        return None

    try:
        _provider_cache = GeminiProvider(api_key=api_key, model=LLM_MODEL)
    except Exception as exc:
        print(f"LLM fallback: could not initialize the Gemini provider ({type(exc).__name__}) - skipping.")
        _provider_cache = None
    return _provider_cache


def _build_prompt(shots_db: dict) -> str:
    lines = [
        "You are a cricket coaching assistant. Identify which cricket batting shot is shown in this image.",
        "Choose exactly one shot_key from this list (use the exact spelling - do not invent new names):",
        "",
    ]
    for name, info in shots_db.items():
        lines.append(f"- {name}: {info.get('summary', '')}")
    lines.append("")
    lines.append(
        'Respond with strict JSON only, no markdown formatting, no extra text: '
        '{"shot_key": "<one of the names above, exactly as written>", '
        '"confidence": <number between 0.0 and 1.0>, "reasoning": "<one short sentence>"}'
    )
    return "\n".join(lines)


def _cache_put(key, value):
    if key in _cache:
        return
    if len(_cache_order) >= _CACHE_MAX_SIZE:
        oldest = _cache_order.pop(0)
        _cache.pop(oldest, None)
    _cache[key] = value
    _cache_order.append(key)


def _parse_response(raw_text: str, shots_db: dict):
    try:
        data = json.loads(raw_text)
        shot_key = data.get("shot_key")
        confidence = float(data.get("confidence", 0.0))
        reasoning = str(data.get("reasoning", ""))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return None

    if shot_key not in shots_db:
        return None
    confidence = max(0.0, min(1.0, confidence))
    return shot_key, confidence, reasoning


def classify_shot_llm(frame, shots_db: dict):
    """
    frame: a clean BGR numpy image (the un-annotated impact frame - callers
    must pass a copy taken *before* the skeleton overlay is drawn on it).
    shots_db: shots.SHOT_DB.

    Returns (shot_key, confidence, reasoning) or None on any failure -
    missing/invalid key, disabled, over budget, network error, rate
    limit, or an unparseable/unvalidatable response. Never raises - this
    is a fallback tier, so any problem here must fall through cleanly to
    the rule-based classifier instead of breaking the analysis.
    """
    global _session_call_count, _session_disabled

    if not LLM_FALLBACK_ENABLED or frame is None:
        return None
    if _session_disabled or _session_call_count >= LLM_MAX_CALLS_PER_SESSION:
        return None

    provider = _get_provider()
    if provider is None:
        return None

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    image_bytes = encoded.tobytes()

    cache_key = hashlib.sha256(image_bytes).hexdigest()
    if cache_key in _cache:
        return _cache[cache_key]

    prompt = _build_prompt(shots_db)

    try:
        _session_call_count += 1
        raw_text = provider.classify(image_bytes, prompt)
    except Exception as exc:
        _handle_provider_error(exc)
        return None

    result = _parse_response(raw_text, shots_db)
    if result is not None:
        _cache_put(cache_key, result)
    return result


def _handle_provider_error(exc: Exception) -> None:
    global _session_disabled
    try:
        from google.genai import errors
        is_rate_limited = isinstance(exc, errors.ClientError) and getattr(exc, "code", None) == 429
    except ImportError:
        is_rate_limited = False

    if is_rate_limited:
        _session_disabled = True
        print("LLM fallback: Gemini free-tier limit hit - skipping LLM fallback for the rest of this session.")
    else:
        print(f"LLM fallback: call failed ({type(exc).__name__}) - falling back to the rule-based classifier.")
