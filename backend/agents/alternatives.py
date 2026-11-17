from __future__ import annotations

import logging
import re
from typing import Any

from backend.schemas.product import Product
from backend.services.product_sources import build_product_service

logger = logging.getLogger("product_alternatives")


def _extract_spec_metrics(item: dict[str, Any] | Product) -> dict[str, Any]:
    """Extract verifiable numeric spec tiers for deterministic comparison."""
    if isinstance(item, Product):
        name = item.name
        desc = item.description
        feats = item.features
        specs = str(item.specifications)
        rating = item.rating or 0.0
    else:
        name = str(item.get("name") or item.get("title") or "")
        desc = str(item.get("description") or "")
        feats = item.get("features") or []
        specs = str(item.get("specifications") or "")
        rating = float(item.get("rating") or 0.0)

    text = f"{name} {desc} {' '.join(feats)} {specs}".lower()

    # RAM (GB)
    ram_gb = 0
    m_ram = re.search(r"\b(\d{1,2})\s*gb\s*(?:ddr[45])?\s*(?:ram|memory)?\b", text)
    if m_ram:
        ram_gb = int(m_ram.group(1))

    # Storage (GB)
    storage_gb = 0
    m_tb = re.search(r"\b(\d)\s*tb\s*(?:ssd|hdd|storage)\b", text)
    if m_tb:
        storage_gb = int(m_tb.group(1)) * 1024
    else:
        m_ssd = re.search(r"\b(\d{2,4})\s*gb\s*(?:ssd|hdd|storage)\b", text)
        if m_ssd:
            storage_gb = int(m_ssd.group(1))

    # CPU Tier
    cpu_tier = 0
    if re.search(r"\b(?:core\s+i9|ryzen\s+9|apple\s+m[34]\s+pro|apple\s+m[34]\s+max)\b", text):
        cpu_tier = 4
    elif re.search(r"\b(?:core\s+i7|ryzen\s+7|apple\s+m[234])\b", text):
        cpu_tier = 3
    elif re.search(r"\b(?:core\s+i5|ryzen\s+5)\b", text):
        cpu_tier = 2
    elif re.search(r"\b(?:core\s+i3|ryzen\s+3)\b", text):
        cpu_tier = 1

    return {
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "cpu_tier": cpu_tier,
        "rating": rating,
    }


def find_product_alternatives_impl(
    original_product: dict[str, Any],
    alternative_type: str = "similar",
    user_requirements: dict[str, Any] | None = None,
    max_results: int = 8,
    service: Any | None = None,
) -> dict[str, Any]:
    """
    Discovers real external alternatives to a product adhering strictly to constraints.
    Supported types: 'similar', 'cheaper', 'different_brands', 'better_specifications', 'budget_friendly'.
    Never invents synthetic alternatives, relaxes budgets, or mislabels unknown prices as cheaper.
    """
    user_reqs = user_requirements or {}
    product_service = service or build_product_service()

    orig_name = str(original_product.get("name") or original_product.get("title") or "").strip()
    orig_brand = str(original_product.get("brand") or "").strip()
    orig_category = str(original_product.get("category") or "").strip()
    orig_subcategory = str(original_product.get("subcategory") or "").strip()
    orig_currency = str(original_product.get("currency") or "INR").upper()

    orig_price = original_product.get("price")
    if orig_price is not None and orig_price != "":
        try:
            orig_price = float(orig_price)
        except (ValueError, TypeError):
            orig_price = None
    else:
        orig_price = None
    orig_price_verified = bool(original_product.get("price_verified", orig_price is not None))
    orig_id = str(original_product.get("product_id") or original_product.get("id") or "")

    preferred_brands = [str(b).strip().lower() for b in user_reqs.get("preferred_brands", []) if str(b).strip()]
    excluded_brands = [str(b).strip().lower() for b in user_reqs.get("excluded_brands", []) if str(b).strip()]
    required_features = [str(f).strip() for f in user_reqs.get("required_features", []) if str(f).strip()]
    max_price = user_reqs.get("max_price")
    if max_price is not None:
        try:
            max_price = float(max_price)
        except (ValueError, TypeError):
            max_price = None

    notes: list[str] = []

    # Alternative type handling
    search_query_parts = []
    target_category = orig_subcategory or orig_category

    if not target_category:
        for cat in ("laptop", "headphones", "earbuds", "smartphone", "suitcase", "backpack", "monitor", "watch", "shoe"):
            if cat in orig_name.lower():
                target_category = cat
                break
        if not target_category:
            target_category = orig_name

    effective_max_price = max_price

    if alternative_type == "cheaper":
        if orig_price is None or not orig_price_verified:
            notes.append("Original product price is unverified; searching for lower-priced alternatives using available budget constraints.")
        else:
            effective_max_price = min(max_price, orig_price - 1.0) if max_price else (orig_price - 1.0)
            notes.append(f"Restricted search to verified options below ₹{int(orig_price):,} in {orig_currency}.")
        search_query_parts.append(f"{target_category} budget affordable")

    elif alternative_type == "budget_friendly":
        if orig_price and orig_price_verified:
            budget_target = orig_price * 0.8
            effective_max_price = min(max_price, budget_target) if max_price else budget_target
            if max_price:
                notes.append(f"Targeting budget-friendly alternatives under ₹{int(effective_max_price):,} while strictly preserving maximum budget constraint of ₹{int(max_price):,}.")
            else:
                notes.append(f"Targeting budget-friendly alternatives under ₹{int(effective_max_price):,}.")
        elif max_price:
            notes.append(f"Targeting budget-friendly alternatives under maximum budget of ₹{int(max_price):,}.")
        search_query_parts.append(f"{target_category} budget value")

    elif alternative_type == "different_brands":
        if orig_brand:
            excluded_brands.append(orig_brand.lower())
            notes.append(f"Excluding original brand '{orig_brand}'.")
        search_query_parts.append(f"{target_category}")

    elif alternative_type == "better_specifications":
        search_query_parts.append(f"{target_category} high performance pro upgrade")
        notes.append("Searching for alternatives with verified enhanced specifications or higher ratings.")

    else:  # 'similar'
        search_query_parts.append(f"{target_category}")
        if required_features:
            search_query_parts.extend(required_features[:2])

    query_str = " ".join(search_query_parts).strip() or target_category

    # Execute search using existing product service
    try:
        search_result = product_service.search(
            query=query_str,
            category=orig_category or None,
            subcategory=orig_subcategory or None,
            max_price=effective_max_price,
            limit=30,
        )
        candidates = search_result.products
    except Exception as exc:
        logger.warning("Error searching for alternatives: %s", exc)
        return {
            "original_product": original_product,
            "alternative_type": alternative_type,
            "alternatives": [],
            "count": 0,
            "search_status": "provider_error",
            "notes": [f"External alternative search failed: {exc}"],
        }

    orig_metrics = _extract_spec_metrics(original_product)

    # Deterministic filtering
    filtered: list[Product] = []
    seen_ids = {orig_id}
    seen_names = {orig_name.lower()}

    for cand in candidates:
        cand_id = cand.product_id
        cand_name = cand.name.lower().strip()
        cand_brand = (cand.brand or "").lower().strip()
        cand_currency = str(cand.currency or "INR").upper()

        # Exclude exact same product
        if cand_id in seen_ids or cand_name in seen_names:
            continue
        seen_ids.add(cand_id)
        seen_names.add(cand_name)

        # Brand exclusion
        if cand_brand and cand_brand in excluded_brands:
            continue

        # Currency consistency
        if cand_currency != orig_currency:
            continue

        # Hard budget enforcement
        if effective_max_price is not None:
            if not cand.price_verified or cand.price is None:
                continue
            if cand.price > effective_max_price:
                continue

        # Special check for 'cheaper': must be strictly less than verified original price
        if alternative_type == "cheaper" and orig_price is not None and orig_price_verified:
            if not cand.price_verified or cand.price is None:
                continue
            if cand.price >= orig_price:
                continue

        # Special check for 'different_brands': must not match original brand
        if alternative_type == "different_brands" and orig_brand:
            if cand_brand and cand_brand == orig_brand.lower():
                continue

        # Special check for 'better_specifications': verify actual spec improvements
        if alternative_type == "better_specifications":
            cand_metrics = _extract_spec_metrics(cand)
            has_better_spec = (
                (cand_metrics["ram_gb"] > 0 and cand_metrics["ram_gb"] > orig_metrics["ram_gb"])
                or (cand_metrics["storage_gb"] > 0 and cand_metrics["storage_gb"] > orig_metrics["storage_gb"])
                or (cand_metrics["cpu_tier"] > 0 and cand_metrics["cpu_tier"] > orig_metrics["cpu_tier"])
                or (cand_metrics["rating"] > 0 and cand_metrics["rating"] > (orig_metrics["rating"] + 0.2))
            )
            if not has_better_spec:
                continue

        filtered.append(cand)

    # Sort candidates: prioritize preferred brands, then ratings, then price
    def alt_sort_key(p: Product) -> tuple[int, float, float]:
        is_pref = 1 if (p.brand or "").lower() in preferred_brands else 0
        rating = p.rating or 3.0
        price_inv = -(p.price or 999999) if alternative_type in ("cheaper", "budget_friendly") else 0
        return (is_pref, rating, price_inv)

    ranked_alts = sorted(filtered, key=alt_sort_key, reverse=True)[:max_results]

    if not ranked_alts:
        notes.append(f"No matching alternatives found satisfying '{alternative_type}' constraints.")
        return {
            "original_product": original_product,
            "alternative_type": alternative_type,
            "alternatives": [],
            "count": 0,
            "search_status": "no_results",
            "notes": notes,
        }

    return {
        "original_product": original_product,
        "alternative_type": alternative_type,
        "alternatives": [p.model_dump(mode="json") for p in ranked_alts],
        "count": len(ranked_alts),
        "search_status": "success",
        "notes": notes,
    }
