from typing import Any

from backend.services.ai import gemini_json, llm_json, openrouter_json


def diagnose_fallback_condition(state: dict[str, Any]) -> tuple[str, str, list[str]]:
    """
    Diagnoses the exact failure condition and produces:
    (failure_code, fallback_message, suggestion_chips)
    """
    constraints = state.get("constraints", {})
    query = state.get("normalized_query", state.get("message", ""))
    retrieval_errors = state.get("retrieval_errors", {})
    provider_status = state.get("provider_status", {})
    raw_candidates = state.get("candidates", [])
    rejection_reasons = state.get("rejection_reasons", {})

    product_type = constraints.get("product_type") or "items"
    budget_max = constraints.get("budget_max")
    preferred_brands = constraints.get("preferred_brands", [])
    excluded_brands = constraints.get("excluded_brands", [])
    preferences = constraints.get("preferences", [])

    # 1. External Provider Failure
    is_provider_error = any(
        "error" in str(err).lower() or "401" in str(err) or "429" in str(err) or "timeout" in str(err)
        for err in retrieval_errors.values()
    ) or any(status not in ("ready", "success_with_results", "success_with_no_results") for status in provider_status.values())

    if is_provider_error and not raw_candidates:
        return (
            "PROVIDER_ERROR",
            "I'm having trouble connecting to live shopping sources right now due to a provider issue. Would you like to retry?",
            ["Retry search"],
        )

    # 2. No products returned from provider
    if not raw_candidates:
        return (
            "NO_PRODUCTS_RETRIEVED",
            f"No live product listings were found for '{query}'. Would you like me to broaden the search terms or explore related categories?",
            ["Broaden search", "Clear filters"],
        )

    # 3. Budget Restriction
    budget_exceeded = rejection_reasons.get("budget_exceeded", 0)
    lowest_seen = rejection_reasons.get("lowest_seen_price")

    if budget_max is not None and budget_exceeded > 0:
        if lowest_seen is not None and lowest_seen < float("inf"):
            lowest_int = int(lowest_seen)
            # Offer realistic target budget
            target_budget = round(lowest_int, -2) if lowest_int > 1000 else lowest_int
            return (
                "BUDGET_RESTRICTION",
                f"I found matching {product_type}, but all available options exceed your budget of ₹{int(budget_max):,}. The lowest verified price found was ₹{lowest_int:,}. Would you like to increase your budget or explore alternative options?",
                [f"Increase budget to ₹{target_budget:,}", "Show alternative options"],
            )
        else:
            return (
                "BUDGET_RESTRICTION",
                f"I couldn't find matching {product_type} within your budget of ₹{int(budget_max):,}. Would you like to adjust your budget or look for refurbished options?",
                ["Increase budget by 20%", "Show affordable options"],
            )

    # 4. Brand Restriction
    brand_excluded = rejection_reasons.get("brand_excluded", 0)
    if (excluded_brands or preferred_brands) and (brand_excluded > 0 or preferred_brands):
        return (
            "BRAND_RESTRICTION",
            f"I couldn't find enough matching {product_type} meeting your brand preferences. Would you like me to include other popular brands?",
            ["Include other brands", "Show all brands"],
        )

    # 5. Overly Restrictive Specifications
    if preferences:
        return (
            "SPECIFICATION_RESTRICTION",
            f"I found products in this category, but none satisfied all your specific requirements ({', '.join(preferences)}). Would you consider relaxing some of these features?",
            ["Relax specifications", "Broaden search"],
        )

    # 6. Unverified Product Information
    unverified = rejection_reasons.get("unverified_price", 0)
    if unverified > 0:
        return (
            "UNVERIFIED_INFORMATION",
            f"Some {product_type} listings were discovered, but their current prices or stock could not be verified against your budget. Would you like me to search across other retailers?",
            ["Search other retailers", "Broaden category"],
        )

    # 7. No Relevant Products (relevance too low)
    return (
        "NO_RELEVANT_PRODUCTS",
        f"I couldn't find sufficiently relevant products for '{query}'. Would you like me to broaden the search to include related alternatives?",
        ["Broaden search", "Search related categories"],
    )


def handle_fallback(state: dict[str, Any]) -> dict[str, Any]:
    constraints = state.get("constraints", {})
    query = state.get("normalized_query", state.get("message", ""))
    
    code, default_msg, suggestions = diagnose_fallback_condition(state)

    # Map failure code to frontend search status
    status_map = {
        "PROVIDER_ERROR": "provider_error",
        "UNVERIFIED_INFORMATION": "no_verified_results",
        "BUDGET_RESTRICTION": "no_results",
        "BRAND_RESTRICTION": "no_results",
        "SPECIFICATION_RESTRICTION": "no_results",
        "NO_PRODUCTS_RETRIEVED": "no_results",
        "NO_RELEVANT_PRODUCTS": "no_results",
    }
    search_status = status_map.get(code, "no_results")

    # Optional LLM refinement for conversational polish
    guidance = openrouter_json(
        f"""Customer shopping request: {query!r}
Constraints: {constraints!r}
Failure diagnosis: {code} ({default_msg})
Suggest one concise, conversational explanation and actionable advice for the customer.
Return JSON only: {{\"message\": \"...\"}}.
Do not claim any product was found if none were verified.""",
        operation="fallback_guidance",
    )
    final_message = (guidance or {}).get("message") or default_msg

    return {
        "fallback_products": [],
        "action": "message",
        "response_message": final_message,
        "search_status": search_status,
        "fallback_reason": code,
        "fallback_suggestions": suggestions,
    }
