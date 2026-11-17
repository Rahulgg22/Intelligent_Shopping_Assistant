from typing import Any


def map_attributes(state: dict[str, Any]) -> dict[str, Any]:
    constraints = state.get("constraints", {})
    query = state.get("normalized_query", state.get("message", ""))

    product_type = constraints.get("product_type") or constraints.get("subcategory") or constraints.get("category") or ""
    recipient = constraints.get("recipient")
    intended_use = constraints.get("intended_use") or constraints.get("activity")
    preferences = constraints.get("preferences", [])
    max_price = constraints.get("budget_max")

    parts = []
    if product_type:
        parts.append(str(product_type))
    if recipient:
        parts.append(f"for {recipient}")
    if intended_use and intended_use not in str(product_type):
        parts.append(f"for {intended_use}")
    for pref in preferences:
        if pref not in str(product_type) and pref not in " ".join(parts):
            parts.append(pref)
    if max_price:
        parts.append(f"under {int(max_price)}")

    constructed = " ".join(parts).strip()
    is_followup = constraints.get("is_followup", False)

    if constructed and (is_followup or len(query.split()) <= 4):
        search_query = constructed
    else:
        search_query = query

    return {
        "mapped_filters": {
            "search_query": search_query,
            "category": constraints.get("category"),
            "subcategory": constraints.get("subcategory"),
            "activity": intended_use,
            "min_price": constraints.get("budget_min"),
            "max_price": constraints.get("budget_max"),
            "preferences": preferences,
            "preferred_brands": constraints.get("preferred_brands", []),
            "excluded_brands": constraints.get("excluded_brands", []),
            "brand_diversity": constraints.get("brand_diversity", False),
            "in_stock_only": "in_stock" in constraints.get("hard_constraints", []),
        }
    }
