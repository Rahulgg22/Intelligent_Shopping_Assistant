from backend.services.ai import gemini_json


def as_list(value):
    if isinstance(value, list):
        return [str(item).lower() for item in value]
    if not value:
        return []
    return [item.strip().lower() for item in str(value).split(";") if item.strip()]


def explain_recommendations(state):
    constraints = state.get("constraints", {})
    activity = constraints.get("activity")
    preferences = as_list(constraints.get("preferences", []))
    budget = constraints.get("budget_max")
    recommendations = []
    catalog_products = state.get("ranked_products", [])[:5]
    llm_prompt = f"""Explain these product recommendations for the shopper.
Shopper request: {state.get('normalized_query', state.get('message', ''))!r}
Constraints: {constraints!r}
Products: {catalog_products!r}
Return JSON only as {{"explanations": [{{"product_id": "...", "explanation": "...", "tradeoffs": ["..."]}}]}}.
Use only catalog fields. Mention honest limitations when supported by the data. Never claim medical suitability."""
    llm_result = gemini_json(llm_prompt)
    llm_explanations = {
        item.get("product_id"): item
        for item in (llm_result or {}).get("explanations", [])
        if isinstance(item, dict)
    }

    for product in catalog_products:
        reasons = []
        use_cases = as_list(product.get("use_cases", ""))
        features = as_list(product.get("features", ""))
        tags = as_list(product.get("tags", ""))
        if activity and any(activity in item for item in use_cases):
            reasons.append(f"it is listed for {activity}")
        for preference in preferences:
            if any(preference in item or item in preference for item in features + tags):
                reasons.append(f"it matches your {preference} preference")
        if budget is not None:
            reasons.append(f"its price is within your budget of {budget:g}")
        recommendations.append({
            **product,
            "explanation": llm_explanations.get(product.get("product_id"), {}).get("explanation")
            or "; ".join(reasons)
            or "it matches the available catalog filters",
            "tradeoffs": llm_explanations.get(product.get("product_id"), {}).get(
                "tradeoffs", product.get("tradeoffs", [])
            ),
        })

    return {
        "recommended_products": recommendations,
        "action": "recommendations",
        "response_message": "Here are the best matches from live shopping results.",
    }