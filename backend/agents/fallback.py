from backend.services.ai import gemini_json


def handle_fallback(state):
    constraints = state.get("constraints", {})
    guidance = gemini_json(
        f"""No verified external products were found for this shopping request: {state.get('normalized_query', '')!r}.
Constraints: {constraints!r}. Suggest one concise explanation of what the shopper can change or try next.
Return JSON only: {{\"message\": \"...\"}}. Do not claim any product is suitable unless retrieved data supports it."""
    )
    return {
        "fallback_products": [],
        "action": "message",
        "response_message": (guidance or {}).get("message")
        or f"No verified products were found for '{state.get('normalized_query', 'that request')}'. Try broadening the category or adjusting your budget, and I can search again.",
    }
