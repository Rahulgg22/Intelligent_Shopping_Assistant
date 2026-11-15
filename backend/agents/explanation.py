import math

LABELS = {"relevance": "relevance to your request", "preference": "matches to your stated preferences",
          "rating": "customer ratings adjusted for review count"}


def _budget_text(constraints):
    lo, hi = constraints.get("budget_min"), constraints.get("budget_max")
    if lo is not None and hi is not None:
        return f"Verified price between INR {float(lo):,.0f} and INR {float(hi):,.0f}."
    if hi is not None:
        return f"Verified price within INR {float(hi):,.0f}."
    if lo is not None:
        return f"Verified price at least INR {float(lo):,.0f}."
    return ""


def explain_recommendations(state):
    ranked = state.get("ranked_products", [])
    constraints = state.get("constraints", {})
    weights = state.get("ranking_weights", {})
    mode = state.get("ranking_mode", "lexical_fallback")
    budget = _budget_text(constraints)
    method = "semantic relevance" if mode == "cross_encoder" else "text matching"
    factors = [f"{method if key == 'relevance' else LABELS[key]} ({weight:.0%})" for key, weight in weights.items()]
    summary = "Results ranked by " + ", ".join(factors) + "." if factors else "Results follow the available recommendation order."
    if budget:
        summary += " All shown products meet your verified price bounds."
    recommendations = []
    for index, product in enumerate(ranked[:5]):
        signals = product.get("rank_signals", {})
        reason = f"Recommended #{index + 1} based on {method}."
        hits = signals.get("preference", {}).get("matched_terms", [])
        if hits:
            reason += " Its listing mentions " + ", ".join(hits) + "."
        title_terms = signals.get("relevance", {}).get("matched_terms", [])
        if title_terms:
            reason += " Title terms matching the search include " + ", ".join(title_terms[:6]) + "."
        comparison = ""
        if index + 1 < len(ranked):
            following = ranked[index + 1]
            if math.isclose(product["rank_score"], following["rank_score"], abs_tol=1e-9, rel_tol=0):
                comparison = "Tied with the next result. Equal scores retain the shortlist order from text matching and source order."
            else:
                differences = {key: signal["contribution"] - following["rank_signals"][key]["contribution"]
                               for key, signal in signals.items()}
                advantage = max(differences, key=differences.get)
                label = method if advantage == "relevance" else LABELS[advantage]
                if advantage == "rating" and signals["rating"].get("neutral"):
                    comparison = "Missing or limited review evidence received a neutral score, which exceeded the next result's rating score."
                else:
                    comparison = f"Its largest scoring advantage over the next result came from {label}."
        elif len(ranked) == 1:
            reason += " This was the only eligible result in the retrieved candidates."
        caveats = []
        if product.get("rating") is None:
            caveats.append("Customer rating data is unavailable.")
        elif signals.get("rating", {}).get("neutral", True):
            caveats.append("Rating evidence is limited or invalid and received a neutral score.")
        availability = "Stock has not been confirmed."
        if product.get("availability_verified") and product.get("availability") == "in_stock":
            availability = "Stock was reported available by the source."
        elif product.get("availability") == "out_of_stock":
            availability = "The source reports this offer out of stock."
        if not product.get("availability_verified"):
            caveats.append(availability)
        details = {"reason": reason, "comparison": comparison, "budget": budget,
                   "availability": availability, "caveats": caveats, "signals": signals}
        recommendations.append({**product, "rank": index + 1, "explanation": reason,
                                "rank_explanation": details, "tradeoffs": caveats})
    return {"recommended_products": recommendations, "action": "recommendations",
            "response_message": "Here are the best matches from live shopping results.",
            "ranking_explanation": {"summary": summary, "source": "deterministic", "method": mode,
                                    "weights": weights, "filters": state.get("filter_stats", {})}}
