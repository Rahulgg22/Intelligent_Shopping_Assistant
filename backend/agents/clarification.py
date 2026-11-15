from backend.services.ai import gemini_json


def _fallback_question(constraints, query=""):
    if not constraints.get("category"):
        if "gift" in query.lower():
            return "What kind of gift would your sister enjoy, such as fashion, books, beauty, electronics, or travel accessories?"
        return "What kind of gear are you looking for, such as shoes, a backpack, or hiking accessories?"
    if not constraints.get("activity"):
        return f"Will the {constraints['category']} be mainly for day walks, hiking, camping, or travel?"
    return "Do you have a maximum budget or a must-have feature I should prioritize?"


def check_clarification(state):
    constraints = state.get("constraints", {})
    if constraints.get("category"):
        return {"needs_clarification": False}

    missing = ["product category"]
    prompt = f"""You are a concise shopping assistant. The customer wants: {state.get('normalized_query', '')!r}.
Known constraints: {constraints!r}. Important missing information: {', '.join(missing)}.
Return JSON only: {{"question": "one natural follow-up question"}}.
Ask about only the most important missing detail; do not ask for information already known."""
    response = gemini_json(prompt)
    return {
        "needs_clarification": True,
        "clarifying_question": response.get("question") if response else _fallback_question(constraints, state.get("normalized_query", "")),
    }