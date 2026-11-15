import json
import logging
import os
import re
from typing import Any

from backend.config import settings

logger = logging.getLogger("gemini")
gemini_state = {"ok": None, "last_error": None, "model": settings.gemini_model}


def gemini_json(prompt: str) -> dict[str, Any] | None:
    api_key = settings.google_api_key.strip()
    if not api_key or api_key == "your_api_key_here":
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = settings.gemini_model
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text or "", flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(text or "{}")
        gemini_state.update({"ok": True, "last_error": None, "model": model})
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        gemini_state.update({"ok": False, "last_error": f"{type(exc).__name__}: {exc}"})
        logger.warning("Gemini call failed: %s", gemini_state["last_error"])
        return None