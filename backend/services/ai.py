import json
import logging
import os
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from backend.config import settings

logger = logging.getLogger("llm")

llm_state: dict[str, Any] = {
    "provider": "openrouter",
    "ok": None,
    "last_error": None,
    "last_category": None,
    "model": settings.openrouter_model,
    "fallback_models": settings.openrouter_fallback_models,
}
# Backward compatibility alias for health checks and existing callers
gemini_state = llm_state

# Account-level permanent error cooldown timestamp
_account_cooldown_until: float = 0.0
_account_cooldown_key: str = ""
_account_cooldown_reason: str = ""


def reset_cooldown() -> None:
    """Resets the account-level error cooldown (useful in tests and manual retries)."""
    global _account_cooldown_until, _account_cooldown_key, _account_cooldown_reason
    _account_cooldown_until = 0.0
    _account_cooldown_key = ""
    _account_cooldown_reason = ""


def _mask_api_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _get_configured_models() -> list[str]:
    """Returns candidate models in priority order starting with primary model."""
    primary = (os.getenv("OPENROUTER_MODEL") or settings.openrouter_model or "google/gemini-2.5-flash").strip()
    fallback_raw = os.getenv("OPENROUTER_FALLBACK_MODELS", settings.openrouter_fallback_models)
    fallbacks = [m.strip() for m in fallback_raw.split(",") if m.strip()]
    
    models = [primary]
    for m in fallbacks:
        if m not in models:
            models.append(m)
    return models


def _clean_json_text(text: str) -> str:
    """Strips markdown code fences and extracts raw JSON object string."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cleaned


def openrouter_json(prompt: str, operation: str = "general") -> dict[str, Any] | None:
    """
    Executes an LLM completion via OpenRouter with automatic multi-model fallback and
    strict classification of permanent vs. transient errors.

    Classification:
      - HTTP 401 / 403: unauthorized / forbidden (permanent: stops candidates, immediate fallback)
      - HTTP 402: payment_required (permanent: no repeated retries, stops candidate models, immediate fallback)
      - HTTP 429 / 5xx / timeouts: rate_limited / server_error (transient: attempts fallback models)
    """
    global _account_cooldown_until, _account_cooldown_key, _account_cooldown_reason

    api_key = (os.getenv("OPENROUTER_API_KEY") or settings.openrouter_api_key or "").strip()
    if not api_key or api_key == "your_openrouter_api_key":
        llm_state.update({
            "ok": False,
            "last_error": "OPENROUTER_API_KEY is not configured",
            "last_category": "missing_credentials",
        })
        logger.info("provider=deterministic operation=%s status=fallback_triggered reason=missing_api_key", operation)
        return None

    # Check if this API key is in permanent error cooldown (e.g. HTTP 402 exhausted credits)
    now = time.time()
    if api_key == _account_cooldown_key and now < _account_cooldown_until:
        logger.info(
            "provider=deterministic operation=%s status=fallback_triggered category=%s reason='%s' (account in cooldown)",
            operation,
            llm_state.get("last_category", "permanent_error"),
            _account_cooldown_reason,
        )
        return None

    endpoint = (os.getenv("OPENROUTER_ENDPOINT") or settings.openrouter_endpoint or "https://openrouter.ai/api/v1").strip()
    timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", settings.openrouter_timeout_seconds))

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
        timeout=timeout,
        default_headers={
            "HTTP-Referer": "https://github.com/shopsmart-ai/shopsmart",
            "X-Title": "PS-12 ShopSmart AI",
        },
    )

    candidate_models = _get_configured_models()
    last_failure_error = None
    last_category = None

    for model_idx, model in enumerate(candidate_models):
        started_at = time.perf_counter()
        try:
            # 1. Attempt structured JSON output
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
            except APIError as api_err:
                err_text = str(api_err).lower()
                status_code = getattr(api_err, "status_code", None)
                if status_code == 400 and ("response_format" in err_text or "json_object" in err_text or "schema" in err_text):
                    logger.debug("Model %s does not support response_format, retrying in standard text mode", model)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                else:
                    raise

            duration = time.perf_counter() - started_at
            raw_text = response.choices[0].message.content or ""
            json_text = _clean_json_text(raw_text)
            parsed = json.loads(json_text)

            if isinstance(parsed, dict):
                logger.info(
                    "provider=openrouter model=%s operation=%s status=success duration=%.2fs",
                    model,
                    operation,
                    duration,
                )
                llm_state.update({
                    "ok": True,
                    "last_error": None,
                    "last_category": None,
                    "model": model,
                })
                return parsed

            logger.warning(
                "provider=openrouter model=%s operation=%s status=invalid_json duration=%.2fs",
                model,
                operation,
                duration,
            )

        except (RateLimitError, APITimeoutError, InternalServerError, APIConnectionError) as transient_err:
            duration = time.perf_counter() - started_at
            status_code = getattr(transient_err, "status_code", "network_error")
            last_category = "rate_limit" if isinstance(transient_err, RateLimitError) else "transient_network"
            last_failure_error = f"{type(transient_err).__name__}: {status_code}"
            logger.info(
                "provider=openrouter operation=%s status=fallback model=%s category=%s error=%s duration=%.2fs",
                operation,
                model,
                last_category,
                status_code,
                duration,
            )
            # Brief backoff before attempting next candidate model
            time.sleep(0.2)
            continue

        except APIStatusError as status_err:
            duration = time.perf_counter() - started_at
            status_code = getattr(status_err, "status_code", 500)
            clean_msg = re.sub(r"key\s*=\s*['\"]?[a-zA-Z0-9_\-]+['\"]?", "key=***", str(status_err))

            # Classify error
            if status_code == 402:
                last_category = "payment_required"
                last_failure_error = f"HTTP 402: Payment Required / Insufficient Credits"
                _account_cooldown_until = time.time() + 30.0  # Cooldown to avoid futile calls
                _account_cooldown_key = api_key
                _account_cooldown_reason = "HTTP 402 Payment Required"
                logger.warning(
                    "provider=openrouter operation=%s model=%s status=failed category=payment_required code=402 error='%s' (stopping candidate models; immediate deterministic fallback)",
                    operation,
                    model,
                    clean_msg,
                )
                break  # Do NOT retry subsequent models under exhausted credits

            elif status_code in (401, 403):
                last_category = "authentication_error"
                last_failure_error = f"HTTP {status_code}: Unauthorized / Forbidden"
                _account_cooldown_until = time.time() + 30.0
                _account_cooldown_key = api_key
                _account_cooldown_reason = f"HTTP {status_code} Auth Failure"
                logger.warning(
                    "provider=openrouter operation=%s model=%s status=failed category=authentication_error code=%s (stopping candidate models)",
                    operation,
                    model,
                    status_code,
                )
                break

            elif status_code in (429, 500, 502, 503, 504):
                last_category = "rate_limit" if status_code == 429 else "server_error"
                last_failure_error = f"APIStatusError: {status_code}"
                logger.info(
                    "provider=openrouter operation=%s status=fallback model=%s category=%s code=%s duration=%.2fs",
                    operation,
                    model,
                    last_category,
                    status_code,
                    duration,
                )
                time.sleep(0.2)
                continue

            else:
                last_category = "client_error"
                last_failure_error = f"APIStatusError: {status_code}"
                logger.warning(
                    "provider=openrouter operation=%s status=failed model=%s category=client_error code=%s error='%s'",
                    operation,
                    model,
                    status_code,
                    clean_msg,
                )
                break

        except APIError as api_err:
            duration = time.perf_counter() - started_at
            status_code = getattr(api_err, "status_code", 500)
            if status_code == 402:
                last_category = "payment_required"
                last_failure_error = f"HTTP 402: Payment Required"
                _account_cooldown_until = time.time() + 30.0
                _account_cooldown_reason = "HTTP 402 Payment Required"
                logger.warning(
                    "provider=openrouter operation=%s model=%s status=failed category=payment_required code=402 (stopping candidate models; immediate deterministic fallback)",
                    operation,
                    model,
                )
                break
            elif status_code in (401, 403):
                last_category = "authentication_error"
                last_failure_error = f"HTTP {status_code}"
                _account_cooldown_until = time.time() + 30.0
                logger.warning(
                    "provider=openrouter operation=%s model=%s status=failed category=auth code=%s",
                    operation,
                    model,
                    status_code,
                )
                break
            elif status_code in (429, 500, 502, 503, 504):
                last_category = "server_error"
                last_failure_error = f"APIError: {status_code}"
                logger.info(
                    "provider=openrouter operation=%s status=fallback model=%s code=%s duration=%.2fs",
                    operation,
                    model,
                    status_code,
                    duration,
                )
                time.sleep(0.2)
                continue
            else:
                last_category = "general_api_error"
                last_failure_error = f"APIError: {status_code}"
                logger.warning(
                    "provider=openrouter operation=%s status=failed model=%s code=%s",
                    operation,
                    model,
                    status_code,
                )
                break

        except (json.JSONDecodeError, ValueError) as json_err:
            duration = time.perf_counter() - started_at
            last_category = "malformed_json"
            last_failure_error = f"JSONDecodeError: {json_err}"
            logger.warning(
                "provider=openrouter operation=%s status=fallback model=%s category=malformed_json duration=%.2fs",
                operation,
                model,
                duration,
            )
            continue

        except Exception as exc:
            duration = time.perf_counter() - started_at
            last_category = "unexpected_error"
            last_failure_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "provider=openrouter operation=%s status=failed model=%s error=%s",
                operation,
                model,
                last_failure_error,
            )
            continue

    # Level B: Fallback to application deterministic logic
    llm_state.update({
        "ok": False,
        "last_error": last_failure_error,
        "last_category": last_category,
    })
    logger.info(
        "provider=deterministic operation=%s status=success category=%s reason=all_models_failed_or_permanent_error",
        operation,
        last_category,
    )
    return None


# Primary unified interface and backward-compatible alias
llm_json = openrouter_json
gemini_json = openrouter_json
