import os


_reranker = None


def _product_text(product):
    return " ".join(
        str(product.get(field, ""))
        for field in ("name", "description", "features", "use_cases", "tags")
    )


def _cross_encoder_scores(query, candidates):
    global _reranker
    try:
        from sentence_transformers import CrossEncoder

        if _reranker is None:
            _reranker = CrossEncoder(
                os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )
        pairs = [(query, _product_text(product)) for product in candidates]
        return [float(score) for score in _reranker.predict(pairs)]
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def split_values(value):
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value]
    if not value:
        return []
    return [item.strip().lower() for item in str(value).split(";") if item.strip()]


def rank_candidates(state):
    constraints = state.get("constraints", {})
    candidates = state.get("candidates", [])
    activity = str(constraints.get("activity", "")).lower()
    preferences = set(split_values(constraints.get("preferences", [])))
    max_price = constraints.get("budget_max")
    min_price = constraints.get("budget_min")
    query = state.get("normalized_query", state.get("message", ""))
    ranked = []

    for product in candidates:
        raw_price = product.get("price")
        price = float(raw_price) if raw_price not in (None, "") else None
        if (max_price is not None or min_price is not None) and price is None:
            continue
        if max_price is not None and price > float(max_price):
            continue
        if min_price is not None and price < float(min_price):
            continue

        use_cases = split_values(product.get("use_cases", ""))
        features = split_values(product.get("features", ""))
        tags = split_values(product.get("tags", ""))
        score = 0.0
        if activity and any(activity in item for item in use_cases):
            score += 0.40
        if preferences:
            score += 0.40 * sum(
                1 for preference in preferences
                if any(preference in item or item in preference for item in features + tags)
            ) / len(preferences)
        rating = product.get("rating")
        score += 0.20 * (float(rating) / 5.0) if rating not in (None, "") else 0
        ranked.append({**product, "fit_score": round(score, 4)})

    ranked.sort(key=lambda product: (product["fit_score"], float(product.get("rating") or 0)), reverse=True)
    cross_scores = _cross_encoder_scores(query, ranked[:30])
    if cross_scores is not None:
        for product, cross_score in zip(ranked[:30], cross_scores):
            product["fit_score"] = round(1 / (1 + __import__("math").exp(-cross_score)), 4)

    ranked.sort(key=lambda product: (product["fit_score"], float(product.get("rating") or 0)), reverse=True)
    return {"ranked_products": ranked}