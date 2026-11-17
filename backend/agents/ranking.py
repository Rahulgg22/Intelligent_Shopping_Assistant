import logging
import math
import os
import re
import time
from typing import Any

from backend.schemas.product import Product

logger = logging.getLogger("ranking")

_reranker = None
_reranker_failed = False
_reranker_init_time = 0.0


def get_ranking_weights() -> dict[str, float]:
    """Configurable weights for the multi-factor ranking algorithm."""
    return {
        "semantic": float(os.getenv("WEIGHT_SEMANTIC", "0.30")),
        "lexical": float(os.getenv("WEIGHT_LEXICAL", "0.20")),
        "attribute": float(os.getenv("WEIGHT_ATTRIBUTE", "0.25")),
        "budget": float(os.getenv("WEIGHT_BUDGET", "0.15")),
        "brand": float(os.getenv("WEIGHT_BRAND", "0.10")),
    }


def _product_text(product: dict[str, Any]) -> str:
    return " ".join(
        str(product.get(field, ""))
        for field in ("name", "description", "features", "use_cases", "tags", "specifications")
    )


def get_cross_encoder():
    """
    Singleton loader for SentenceTransformers CrossEncoder.
    Ensures model is initialized only once and reused across all ranking requests.
    Prevents repeated downloads, weights reloading, and network timeouts.
    """
    global _reranker, _reranker_failed, _reranker_init_time
    if _reranker is not None:
        return _reranker, 0.0
    if _reranker_failed:
        return None, 0.0

    t0 = time.perf_counter()
    try:
        # Enforce offline cache loading first to prevent 30s HuggingFace timeout backoff
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        except Exception:
            pass

        from sentence_transformers import CrossEncoder

        model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            _reranker = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            _reranker = CrossEncoder(model_name)

        _reranker_init_time = time.perf_counter() - t0
        logger.info(
            "ranking_timing phase=model_initialization took=%.4fs model=%s reused=False",
            _reranker_init_time,
            model_name,
        )
        return _reranker, _reranker_init_time
    except Exception as exc:
        _reranker_failed = True
        logger.warning("ranking_timing phase=model_initialization status=failed error=%s", exc)
        return None, time.perf_counter() - t0


def _cross_encoder_scores(query: str, candidates: list[dict[str, Any]]) -> list[float] | None:
    """
    Computes CrossEncoder scores using the cached singleton instance.
    Returns list of scores or None if unavailable.
    """
    reranker, init_time = get_cross_encoder()
    if reranker is None or not candidates:
        return None

    t_inf_start = time.perf_counter()
    try:
        pairs = [(query, _product_text(product)) for product in candidates]
        raw_scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        scores = [float(score) for score in raw_scores]
        inf_time = time.perf_counter() - t_inf_start
        logger.info(
            "ranking_timing phase=cross_encoder_inference took=%.4fs pairs_scored=%d init_time=%.4fs",
            inf_time,
            len(pairs),
            init_time,
        )
        return scores
    except Exception as exc:
        logger.warning("ranking_timing phase=cross_encoder_inference status=failed error=%s", exc)
        return None


def split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if not value:
        return []
    return [item.strip().lower() for item in str(value).split(";") if item.strip()]


def _lexical_similarity(query_terms: set[str], product: dict[str, Any]) -> float:
    if not query_terms:
        return 0.5
    title = str(product.get("name", "")).lower()
    body = _product_text(product).lower()
    
    title_matches = sum(1 for term in query_terms if re.search(rf"\b{re.escape(term)}\b", title))
    body_matches = sum(1 for term in query_terms if re.search(rf"\b{re.escape(term)}\b", body))
    
    score = (title_matches * 2.0 + body_matches * 1.0) / (len(query_terms) * 2.0)
    return min(1.0, max(0.0, score))


def _semantic_overlap_score(query: str, product: dict[str, Any]) -> float:
    text = _product_text(product).lower()
    q_words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
    if not q_words:
        return 0.5
    matches = sum(1 for w in q_words if w in text)
    return min(1.0, matches / len(q_words))


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicates exact duplicate listings across search sources,
    while preserving separate retailer offers with different prices or merchants.
    """
    unique_items = []
    seen_keys = set()

    for item in candidates:
        url = str(item.get("product_url") or "").strip().lower()
        title = re.sub(r"[^\w\s]", "", str(item.get("name") or "").lower()).strip()
        source = str(item.get("source_name") or "unknown").strip().lower()
        price = item.get("price")

        # Key captures exact merchant offer
        if url and "http" in url:
            key = f"url:{url}"
        else:
            key = f"offer:{source}:{title}:{price}"

        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_items.append(item)

    return unique_items


def rank_candidates(state: dict[str, Any]) -> dict[str, Any]:
    t_total_start = time.perf_counter()
    constraints = state.get("constraints", {})
    raw_candidates = state.get("candidates", [])
    candidates = _deduplicate_candidates(raw_candidates)

    weights = get_ranking_weights()
    w_sem = weights["semantic"]
    w_lex = weights["lexical"]
    w_att = weights["attribute"]
    w_bud = weights["budget"]
    w_brd = weights["brand"]

    query = state.get("normalized_query", state.get("message", "")).strip()
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - {
        "a", "an", "the", "in", "for", "with", "under", "below", "above", "to",
        "find", "show", "suggest", "recommend", "me", "i", "need", "want"
    }

    product_type = str(constraints.get("product_type") or "").strip().lower()
    category = str(constraints.get("category") or "").strip().lower()
    subcategory = str(constraints.get("subcategory") or "").strip().lower()
    intended_use = str(constraints.get("intended_use") or constraints.get("activity") or "").strip().lower()
    preferences = set(split_values(constraints.get("preferences", [])))
    preferred_brands = set(split_values(constraints.get("preferred_brands", [])))
    excluded_brands = set(split_values(constraints.get("excluded_brands", [])))
    brand_diversity = bool(constraints.get("brand_diversity"))
    cheaper = bool(constraints.get("cheaper"))
    max_price = constraints.get("budget_max")
    min_price = constraints.get("budget_min")

    rejection_reasons = {
        "total_candidates": len(raw_candidates),
        "deduplicated_candidates": len(candidates),
        "budget_exceeded": 0,
        "unverified_price": 0,
        "brand_excluded": 0,
        "specification_unmatched": 0,
        "lowest_seen_price": float("inf"),
        "unmatched_specifications": list(preferences),
    }

    t_filter_start = time.perf_counter()
    filtered_candidates = []

    # 1. Deterministic Hard Constraints Enforcement
    for product in candidates:
        raw_price = product.get("price")
        price = float(raw_price) if raw_price not in (None, "") else None
        brand = str(product.get("brand") or "").strip().lower()
        name_lower = str(product.get("name") or "").lower()

        # Track lowest observed price for intelligent fallback
        if price is not None and price < rejection_reasons["lowest_seen_price"]:
            rejection_reasons["lowest_seen_price"] = price

        # Hard brand exclusion
        if excluded_brands and (brand in excluded_brands or any(ex in name_lower for ex in excluded_brands)):
            rejection_reasons["brand_excluded"] += 1
            continue

        # Hard budget constraints
        if (max_price is not None or min_price is not None) and price is None:
            rejection_reasons["unverified_price"] += 1
            continue
        if max_price is not None and price > float(max_price):
            rejection_reasons["budget_exceeded"] += 1
            continue
        if min_price is not None and price < float(min_price):
            rejection_reasons["budget_exceeded"] += 1
            continue

        # Hard in-stock constraint
        if "in_stock" in constraints.get("hard_constraints", []):
            if product.get("availability") == "out_of_stock":
                continue

        filtered_candidates.append(product)

    # 2. Multi-Factor Scoring & Evidence Gathering
    scored = []
    for product in filtered_candidates:
        brand = str(product.get("brand") or "").strip().lower()
        name = str(product.get("name") or "")
        name_lower = name.lower()
        price = float(product.get("price")) if product.get("price") is not None else None
        features = split_values(product.get("features", []))
        use_cases = split_values(product.get("use_cases", []))
        tags = split_values(product.get("tags", []))
        specs_text = " ".join(features + use_cases + tags + [product.get("description", "")]).lower()

        matched_reqs = []
        benefits = []
        limitations = []

        # A. Lexical Relevance (20%)
        s_lex = _lexical_similarity(query_terms, product)

        # B. Semantic Relevance (30%)
        s_sem = _semantic_overlap_score(query, product)

        # C. Attribute Matching (25%)
        att_score = 0.0
        # Category / product type match
        if product_type and (product_type in name_lower or product_type in str(product.get("category", "")).lower()):
            att_score += 0.35
            matched_reqs.append(product_type.title())
        elif category and (category in str(product.get("category", "")).lower()):
            att_score += 0.20
            matched_reqs.append(category.title())

        # Intended use match
        if intended_use and (any(intended_use in item for item in use_cases) or intended_use in specs_text or intended_use in name_lower):
            att_score += 0.35
            matched_reqs.append(f"For {intended_use.title()}")
            benefits.append(f"Optimized for {intended_use}")

        # Specifications / Preferences match
        if preferences:
            matched_specs = [
                pref for pref in preferences
                if any(pref in item for item in features + tags) or pref in specs_text or pref in name_lower
            ]
            if matched_specs:
                att_score += 0.30 * (len(matched_specs) / len(preferences))
                for spec in matched_specs:
                    feat_match = next((f for f in features if spec in f.lower()), None)
                    display_spec = feat_match if feat_match else (spec.upper() if len(spec) <= 4 else spec.title())
                    matched_reqs.append(display_spec)
                    benefits.append(f"Features {display_spec}")
        else:
            att_score += 0.30

        s_att = min(1.0, att_score)

        # D. Budget Compatibility (15%)
        s_bud = 0.5
        if max_price is not None and price is not None:
            matched_reqs.append(f"Within budget (₹{int(price):,})")
            if cheaper:
                # Cheaper is better
                s_bud = max(0.0, min(1.0, 1.0 - (price / float(max_price))))
            else:
                # Proximity to good tier without exceeding
                ratio = price / float(max_price)
                s_bud = min(1.0, 0.5 + (0.5 * ratio))
        elif price is not None:
            s_bud = 0.7

        # E. Brand Preference (10%)
        s_brd = 0.5
        if preferred_brands:
            if brand in preferred_brands or any(pb in name_lower for pb in preferred_brands):
                s_brd = 1.0
                matched_brand_display = brand.upper() if len(brand) <= 3 else brand.title()
                matched_reqs.append(f"Preferred brand ({matched_brand_display})")
            else:
                s_brd = 0.1
        elif brand:
            s_brd = 0.6

        # F. Ratings & Verification (Bonus)
        rating = product.get("rating")
        rating_bonus = 0.0
        if rating not in (None, ""):
            r_val = float(rating)
            if r_val >= 4.0:
                rating_bonus = 0.05 * (r_val / 5.0)
                benefits.append(f"High customer rating ({r_val}★)")
        else:
            limitations.append("Customer ratings not available yet")

        if product.get("price_verified") is not True:
            limitations.append("Price not independently verified on merchant page")
        if product.get("availability") == "unknown":
            limitations.append("Stock availability unconfirmed")

        # Top product features as benefits
        for feat in features[:3]:
            if feat not in " ".join(benefits).lower():
                benefits.append(feat.capitalize())

        # Weighted Composite Score
        composite_score = (
            (w_sem * s_sem) +
            (w_lex * s_lex) +
            (w_att * s_att) +
            (w_bud * s_bud) +
            (w_brd * s_brd) +
            rating_bonus
        )

        scored.append({
            **product,
            "fit_score": round(composite_score, 4),
            "_subscores": {
                "semantic": round(s_sem, 3),
                "lexical": round(s_lex, 3),
                "attribute": round(s_att, 3),
                "budget": round(s_bud, 3),
                "brand": round(s_brd, 3),
            },
            "matched_requirements": list(dict.fromkeys(matched_reqs)),
            "benefits": list(dict.fromkeys(benefits))[:4],
            "limitations": list(dict.fromkeys(limitations)),
        })

    t_filter_end = time.perf_counter()
    logger.info(
        "ranking_timing phase=candidate_filtering took=%.4fs candidates_in=%d candidates_out=%d",
        t_filter_end - t_filter_start,
        len(candidates),
        len(filtered_candidates),
    )

    # Sort initially by composite fit score and rating
    scored.sort(key=lambda p: (p["fit_score"], float(p.get("rating") or 0)), reverse=True)

    # 3. Neural Cross-Encoder Reranking (Batched & Pruned to Top K)
    top_k = min(len(scored), int(os.getenv("RERANKER_TOP_K", "10")))
    if top_k > 0:
        cross_scores = _cross_encoder_scores(query, scored[:top_k])
        if cross_scores is not None:
            for product, cross_score in zip(scored[:top_k], cross_scores):
                # Sigmoid scaling
                neural_score = 1.0 / (1.0 + math.exp(-cross_score))
                # Blended with deterministic multi-factor score (70% deterministic + 30% neural)
                product["fit_score"] = round(0.70 * product["fit_score"] + 0.30 * neural_score, 4)
            scored.sort(key=lambda p: (p["fit_score"], float(p.get("rating") or 0)), reverse=True)

    # 4. Final Ranking & Brand Diversity Optimization
    t_final_start = time.perf_counter()
    if (brand_diversity or True) and len(scored) > 2:
        seen_brands = set()
        diverse = []
        remainder = []
        for p in scored:
            p_brand = str(p.get("brand") or p.get("source_name") or "other").lower()
            if p_brand not in seen_brands:
                seen_brands.add(p_brand)
                diverse.append(p)
            else:
                remainder.append(p)
        scored = diverse + remainder

    # 5. Assign Explainable Ranks and Comparative Notes
    for idx, p in enumerate(scored, start=1):
        p["rank"] = idx
        if idx == 1:
            p["comparison_note"] = "Top recommendation: highest overall match for your requirements and budget."
        elif idx == 2:
            p["comparison_note"] = "Strong runner-up: excellent alternative with balanced specifications."
        else:
            p["comparison_note"] = f"Ranked #{idx} based on price and available feature compatibility."

    t_final_end = time.perf_counter()
    logger.info(
        "ranking_timing phase=final_ranking took=%.4fs products_ranked=%d",
        t_final_end - t_final_start,
        len(scored),
    )
    logger.info("ranking_timing phase=overall took=%.4fs", time.perf_counter() - t_total_start)

    return {
        "ranked_products": scored,
        "rejection_reasons": rejection_reasons,
    }
