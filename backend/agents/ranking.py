import logging
import math
import os
import re

logger = logging.getLogger(__name__)
_reranker = None
MAX_CANDIDATES = 30
BASE_WEIGHTS = {"relevance": 0.70, "preference": 0.20, "rating": 0.10}


def split_values(value):
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [item.strip().lower() for item in str(value or "").split(";") if item.strip()]


def _product_text(product):
    # Match evidence uses listing fields, excluding explanations from earlier searches.
    parts = [str(product.get(field) or "") for field in ("name", "brand", "description")]
    for field in ("features", "use_cases", "tags"):
        parts.extend(split_values(product.get(field)))
    return " ".join(parts)


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _cross_encoder_scores(query, candidates):
    global _reranker
    try:
        from sentence_transformers import CrossEncoder
        if _reranker is None:
            _reranker = CrossEncoder(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        return [float(score) for score in _reranker.predict([(query, _product_text(p)) for p in candidates])]
    except Exception:
        logger.warning("CrossEncoder unavailable, using text matching", exc_info=True)
        return None


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def rank_candidates(state):
    constraints = state.get("constraints", {})
    lo, hi = _number(constraints.get("budget_min")), _number(constraints.get("budget_max"))
    preferences = list(dict.fromkeys(split_values(constraints.get("preferences"))))
    query_parts = [constraints.get(key) for key in ("product_type", "subcategory", "activity")]
    query = " ".join(dict.fromkeys(str(part) for part in query_parts if part))
    query = query or state.get("normalized_query", state.get("message", ""))
    query_tokens = _tokens(query)
    stats = {"considered": len(state.get("candidates", [])), "unverified_price": 0,
             "over_budget": 0, "under_min": 0, "unconfirmed_stock": 0, "outside_shortlist": 0}
    eligible = []
    for product in state.get("candidates", []):
        price = _number(product.get("price"))
        if (lo is not None or hi is not None) and (price is None or not product.get("price_verified")):
            stats["unverified_price"] += 1
            continue
        if hi is not None and price > hi:
            stats["over_budget"] += 1
            continue
        if lo is not None and price < lo:
            stats["under_min"] += 1
            continue
        if "in_stock" in constraints.get("hard_constraints", []) and not (
            product.get("availability") == "in_stock" and product.get("availability_verified")
        ):
            stats["unconfirmed_stock"] += 1
            continue
        eligible.append(product)
    # Stable lexical preselection bounds model work and preserves source order for ties.
    def lexical(product):
        return len(query_tokens & _tokens(_product_text(product))) / max(1, len(query_tokens))
    eligible.sort(key=lexical, reverse=True)
    shortlist = eligible[:MAX_CANDIDATES]
    stats["outside_shortlist"] = len(eligible) - len(shortlist)
    raw_scores = _cross_encoder_scores(query, shortlist) if shortlist else None
    model_ok = raw_scores is not None and len(raw_scores) == len(shortlist) and all(math.isfinite(s) for s in raw_scores)
    mode = "cross_encoder" if model_ok else "lexical_fallback"
    active = ["relevance", "rating"] + (["preference"] if preferences else [])
    total_weight = sum(BASE_WEIGHTS[key] for key in active)
    weights = {key: BASE_WEIGHTS[key] / total_weight for key in active}
    ranked = []
    for index, product in enumerate(shortlist):
        text = _product_text(product)
        hits = [term for term in preferences if re.search(r"\b" + re.escape(term) + r"\b", text, re.I)]
        rating, reviews = _number(product.get("rating")), _number(product.get("review_count"))
        rating_known = rating is not None and 0 <= rating <= 5 and reviews is not None and reviews >= 5
        rating_value = 0.5 + (rating / 5 - 0.5) * min(1, reviews / 20) if rating_known else 0.5
        relevance = 1 / (1 + math.exp(-max(-30, min(30, raw_scores[index])))) if model_ok else lexical(product)
        values = {"relevance": relevance, "rating": rating_value}
        if preferences:
            values["preference"] = len(hits) / len(preferences)
        signals = {key: {"value": values[key], "weight": weights[key],
                         "contribution": values[key] * weights[key]} for key in active}
        signals["relevance"]["matched_terms"] = sorted(query_tokens & _tokens(str(product.get("name") or "")))
        signals["rating"]["neutral"] = not rating_known
        if preferences:
            signals["preference"]["matched_terms"] = hits
        score = sum(signal["contribution"] for signal in signals.values())
        ranked.append({**product, "fit_score": score, "rank_score": score, "rank_signals": signals})
    # Equal scores retain shortlist order. This rule is also reported by the explainer.
    ranked.sort(key=lambda product: product["rank_score"], reverse=True)
    return {"ranked_products": ranked, "ranking_mode": mode, "ranking_query": query,
            "filter_stats": stats, "ranking_weights": weights}
