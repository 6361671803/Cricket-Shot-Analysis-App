"""LLM-based shot classification fallback (tier 2 of the classification
chain: ML -> LLM -> rule-based).

Used only when classify_shot_ml() didn't produce a confident result (no
trained model, or low confidence). Tries the first provider in
LLM_PROVIDER_ORDER (default: gemini); a later provider (e.g. openrouter)
is only ever attempted as a backup for the specific case where the one
before it has exhausted its own daily quota/rate limit this session -
any other failure (no key configured, network error, unparseable
response) stops the chain immediately and falls through to the
rule-based classifier, exactly as it did before this tier existed. With
no provider configured at all, or LLM_FALLBACK_ENABLED=false, this tier
is a silent no-op.

Real trade-off worth naming plainly: using this sends the impact photo
to a third-party API (Google, and/or whichever model OpenRouter routes
the request to).
"""

import hashlib
import json
import os

import cv2
from dotenv import load_dotenv

load_dotenv()

# LLM_PROVIDER (singular) is kept for backward compatibility with existing
# .env files that only set one provider. LLM_PROVIDER_ORDER is the new,
# preferred setting - a comma-separated fallback order. If it's not set,
# it falls back to whatever LLM_PROVIDER says (so an existing .env with
# just LLM_PROVIDER=gemini keeps working unchanged).
_default_order = os.environ.get("LLM_PROVIDER", "gemini")
LLM_PROVIDER_ORDER = [
    p.strip() for p in os.environ.get("LLM_PROVIDER_ORDER", _default_order).split(",") if p.strip()
]

LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite")
# minimax/minimax-m3:free is confirmed working as of September 2026 - tested
# directly against the live OpenRouter API with this project's actual
# classification prompt, returned a valid, parseable JSON response.
# meta-llama/llama-3.2-11b-vision-instruct:free (the original default here)
# was tested the same way and turned out to have been deprecated/removed
# ("No endpoints found"); google/gemma-4-31b-it:free and two other
# candidates were also tested but hit OpenRouter's free-tier daily cap (50
# requests/day per free model unless the account adds credits, which raises
# it to 1000/day). Free-tier model availability changes over time (the same
# happened to this project's Gemini model choice - see the LLM_MODEL comment
# in .env.example) - check https://openrouter.ai/models (filter: image
# input) if this one stops working.
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m3:free").strip()

LLM_FALLBACK_ENABLED = os.environ.get("LLM_FALLBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
LLM_MAX_CALLS_PER_SESSION = int(os.environ.get("LLM_MAX_CALLS_PER_SESSION", "400"))

_CACHE_MAX_SIZE = 100
_cache = {}
_cache_order = []

# Per-provider session state, keyed by provider name (e.g. "gemini",
# "openrouter") - kept separate per provider so one provider hitting its
# rate limit doesn't block a different, still-working provider from
# being tried.
_session_call_count = {}
_session_disabled = {}
_warned_missing_key = {}


class LLMProvider:
    """Common interface every LLM provider implements."""

    def classify(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Free tier, no credit card required. See .env.example for GEMINI_API_KEY."""

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


class OpenRouterProvider(LLMProvider):
    """
    Backup provider - tried only when Gemini has specifically exhausted its
    own daily quota/rate limit for this session (not for a missing key,
    network error, or any other Gemini failure - see classify_shot_llm()).
    Uses OpenRouter's OpenAI-compatible
    chat-completions endpoint (https://openrouter.ai/docs/api-reference/
    chat-completion): POST https://openrouter.ai/api/v1/chat/completions,
    image sent as an image_url content part, JSON-only output requested
    via response_format={"type": "json_object"}.

    Requires OPENROUTER_API_KEY and OPENROUTER_MODEL (see .env.example -
    OpenRouter's free-tier vision-capable model list changes over time,
    so no default model is assumed here; pick a current one from
    https://openrouter.ai/models (filter: image input) and verify it
    actually accepts image input before relying on it).
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def classify(self, image_bytes: bytes, prompt: str) -> str:
        import base64
        import requests

        b64_image = base64.b64encode(image_bytes).decode("ascii")
        response = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    ],
                }],
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicProvider(LLMProvider):
    """TODO: implement against the Anthropic Messages API if a paid key is ever added."""

    def classify(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError("AnthropicProvider is a stub - not implemented yet.")


class OpenAIProvider(LLMProvider):
    """TODO: implement against the OpenAI API if a paid key is ever added."""

    def classify(self, image_bytes: bytes, prompt: str) -> str:
        raise NotImplementedError("OpenAIProvider is a stub - not implemented yet.")


_PROVIDER_BUILDERS = {
    "gemini": lambda: _build_gemini_provider(),
    "openrouter": lambda: _build_openrouter_provider(),
}

_provider_cache = {}
_provider_load_attempted = {}


def _build_gemini_provider():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, "GEMINI_API_KEY not set - skipping (see .env.example to add one)."
    try:
        return GeminiProvider(api_key=api_key, model=LLM_MODEL), None
    except Exception as exc:
        return None, f"could not initialize the Gemini provider ({type(exc).__name__}) - skipping."


def _build_openrouter_provider():
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None, "OPENROUTER_API_KEY not set - skipping (see .env.example to add one)."
    if not OPENROUTER_MODEL:
        return None, "OPENROUTER_MODEL not set - skipping (see .env.example; pick a current vision model from openrouter.ai/models)."
    try:
        return OpenRouterProvider(api_key=api_key, model=OPENROUTER_MODEL), None
    except Exception as exc:
        return None, f"could not initialize the OpenRouter provider ({type(exc).__name__}) - skipping."


def _get_provider(name: str):
    """Lazily builds and caches the named provider. Never raises - returns
    None (logging once per provider) for any config/init problem, or for
    an unknown provider name."""
    if name in _provider_load_attempted:
        return _provider_cache.get(name)
    _provider_load_attempted[name] = True

    builder = _PROVIDER_BUILDERS.get(name)
    if builder is None:
        print(f"LLM fallback: provider '{name}' has no implementation yet - skipping.")
        _provider_cache[name] = None
        return None

    provider, warning = builder()
    if warning and not _warned_missing_key.get(name):
        print(f"LLM fallback ({name}): {warning}")
        _warned_missing_key[name] = True
    _provider_cache[name] = provider
    return provider


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

    Only the first provider in LLM_PROVIDER_ORDER (default: "gemini") is
    tried under normal conditions. A later provider (e.g. "openrouter") is
    only ever attempted as a backup for the *specific* case where the
    provider before it in the chain has exhausted its own quota/rate
    limit for this session (an HTTP 429 / RESOURCE_EXHAUSTED response, or
    this app's own LLM_MAX_CALLS_PER_SESSION soft cap, which exists
    specifically as an early proxy for that same daily quota). Any other
    failure mode - no API key configured, a network error, an
    unparseable/unvalidatable response - stops the chain immediately and
    falls straight through to the rule-based classifier; it does not
    cascade to a backup provider. Never raises - this is a fallback tier,
    so any problem here must fall through cleanly to the rule-based
    classifier instead of breaking the analysis.
    """
    if not LLM_FALLBACK_ENABLED or frame is None:
        return None

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    image_bytes = encoded.tobytes()

    cache_key = hashlib.sha256(image_bytes).hexdigest()
    if cache_key in _cache:
        return _cache[cache_key]

    prompt = _build_prompt(shots_db)

    for i, name in enumerate(LLM_PROVIDER_ORDER):
        if i > 0 and not _session_disabled.get(LLM_PROVIDER_ORDER[i - 1]):
            # The previous provider in the chain did not fail due to its
            # own quota being exhausted - a backup provider only ever
            # engages for that specific case, so stop here instead of
            # cascading further.
            break

        if _session_disabled.get(name):
            continue  # already known quota-exhausted this session - go straight to the backup

        if _session_call_count.get(name, 0) >= LLM_MAX_CALLS_PER_SESSION:
            _session_disabled[name] = True  # treat our own soft cap as quota-exhausted too
            print(f"LLM fallback: {name}'s session call budget reached - treating as quota-exhausted.")
            continue

        provider = _get_provider(name)
        if provider is None:
            break  # not configured / no implementation - not a quota condition, stop here

        try:
            _session_call_count[name] = _session_call_count.get(name, 0) + 1
            raw_text = provider.classify(image_bytes, prompt)
        except Exception as exc:
            _handle_provider_error(name, exc)
            if not _session_disabled.get(name):
                break  # a non-quota failure - stop the chain here, fall through to rule-based
            continue  # quota exhausted - the next provider in the chain, if any, will be tried

        result = _parse_response(raw_text, shots_db)
        if result is not None:
            _cache_put(cache_key, result)
            return result
        break  # got a response but couldn't parse/validate it - not a quota issue, stop here

    return None


def _handle_provider_error(name: str, exc: Exception) -> None:
    is_rate_limited = False
    if name == "gemini":
        try:
            from google.genai import errors
            is_rate_limited = isinstance(exc, errors.ClientError) and getattr(exc, "code", None) == 429
        except ImportError:
            is_rate_limited = False
    elif name == "openrouter":
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        is_rate_limited = status_code == 429

    if is_rate_limited:
        _session_disabled[name] = True
        print(f"LLM fallback: {name} free-tier limit hit - skipping {name} for the rest of this session.")
    else:
        print(f"LLM fallback: {name} call failed ({type(exc).__name__}) - not a quota issue, "
              f"falling through to the rule-based classifier.")
