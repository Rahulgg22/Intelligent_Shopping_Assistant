from __future__ import annotations

import hashlib
import re
from typing import Any


def _stable_id(val: str) -> str:
    return hashlib.sha1(val.encode("utf-8")).hexdigest()[:16]


EXCHANGE_RATES_TO_INR: dict[str, float] = {
    "INR": 1.0,
    "USD": 86.5,
    "EUR": 94.0,
    "GBP": 110.0,
    "AED": 23.5,
    "SGD": 65.0,
    "CAD": 63.0,
    "AUD": 56.5,
}

SPEC_DIMENSION_KEYWORDS = {
    "Processor": [r"\b(?:core\s+i[3579]|ryzen\s+[3579]|m[1234]|intel|amd|snapdragon|tensor|mediatek|processor|cpu)\b"],
    "RAM / Memory": [r"\b(\d{1,2}\s*gb(?:\s*ddr[45])?)\s*(?:ram|memory)?\b"],
    "Storage / SSD": [r"\b(\d{2,4}\s*(?:gb|tb)\s*(?:ssd|hdd|nvme|emmc|storage))\b"],
    "Display / Screen": [r"\b(\d{1,2}(?:\.\d+)?\s*(?:inch|\"|cm)|fhd|oled|amoled|ips|retina|120hz|144hz|60hz|4k)\b"],
    "Battery / Runtime": [r"\b(\d{2,5}\s*mah|\d{1,2}\s*(?:hours?|hrs?)\s*(?:battery|runtime|playback))\b"],
    "Weight": [r"\b(\d+(?:\.\d+)?\s*(?:kg|grams?|g))\b"],
    "Warranty": [r"\b(\d{1,2}\s*(?:months?|years?)\s*warranty)\b"],
    "Noise Cancellation": [r"\b(anc|active\s+noise\s+cancellation|noise\s+cancelling)\b"],
    "Capacity": [r"\b(\d{1,3}\s*(?:l|litres?|liters?))\b"],
    "Operating System": [r"\b(windows\s*11|windows\s*10|macos|chrome\s*os|linux|android|ios)\b"],
}


def _normalize_product_dict(p: dict[str, Any]) -> dict[str, Any]:
    name = str(p.get("name") or p.get("title") or "").strip()
    pid = str(p.get("product_id") or p.get("id") or (_stable_id(name) if name else "prod_unknown"))
    brand = str(p.get("brand") or "").strip()
    price = p.get("price")
    if price is not None and price != "":
        try:
            price = float(price)
        except (ValueError, TypeError):
            price = None
    else:
        price = None

    currency = str(p.get("currency") or "INR").upper()
    rate = EXCHANGE_RATES_TO_INR.get(currency)
    price_inr = round(price * rate, 2) if (price is not None and rate is not None) else None

    price_verified = bool(p.get("price_verified", price is not None))
    product_url = p.get("product_url") or p.get("url") or None
    image_url = p.get("image_url") or None
    rating = p.get("rating")
    if rating is not None and rating != "":
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            rating = None
    else:
        rating = None

    review_count = p.get("review_count")
    if review_count is not None and review_count != "":
        try:
            review_count = int(review_count)
        except (ValueError, TypeError):
            review_count = None
    else:
        review_count = None

    availability = str(p.get("availability") or "unknown")
    source_name = str(p.get("source_name") or "external")

    # Specifications
    specs = p.get("specifications") or {}
    spec_dict: dict[str, str] = {}
    if isinstance(specs, dict):
        spec_dict = {str(k).strip(): str(v).strip() for k, v in specs.items() if str(k).strip()}
    elif isinstance(specs, str) and specs.strip():
        for chunk in specs.split(";"):
            if ":" in chunk:
                k, v = chunk.split(":", 1)
                spec_dict[k.strip()] = v.strip()
            elif chunk.strip():
                spec_dict[chunk.strip()] = "Yes"

    features = p.get("features") or []
    if isinstance(features, str):
        features = [f.strip() for f in features.split(";") if f.strip()]
    elif isinstance(features, list):
        features = [str(f).strip() for f in features if str(f).strip()]

    description = str(p.get("description") or "")
    use_cases = p.get("use_cases") or []
    if isinstance(use_cases, str):
        use_cases = [u.strip() for u in use_cases.split(";") if u.strip()]
    elif isinstance(use_cases, list):
        use_cases = [str(u).strip() for u in use_cases if str(u).strip()]

    return {
        "product_id": pid,
        "name": name,
        "brand": brand,
        "price": price,
        "price_inr": price_inr,
        "currency": currency,
        "currency_compatible": rate is not None,
        "price_verified": price_verified,
        "product_url": product_url,
        "image_url": image_url,
        "rating": rating,
        "review_count": review_count,
        "availability": availability,
        "source_name": source_name,
        "specifications": spec_dict,
        "features": features,
        "description": description,
        "use_cases": use_cases,
    }


def _extract_available_and_missing_specs(
    product: dict[str, Any],
    all_category_dimensions: set[str],
) -> tuple[dict[str, str], list[str]]:
    searchable_text = " ".join([
        product["name"],
        product["brand"],
        product["description"],
        " ".join(f"{k}: {v}" for k, v in product["specifications"].items()),
        " ".join(product["features"]),
        " ".join(product["use_cases"]),
    ]).lower()

    available: dict[str, str] = {}

    for k, v in product["specifications"].items():
        available[k] = v

    for dim, patterns in SPEC_DIMENSION_KEYWORDS.items():
        if dim in available:
            continue
        for pat in patterns:
            match = re.search(pat, searchable_text, re.IGNORECASE)
            if match:
                available[dim] = match.group(0).strip()
                break

    missing: list[str] = []
    for dim in sorted(all_category_dimensions):
        if dim not in available:
            missing.append(f"{dim} (not specified in listing)")

    return available, missing


def _evaluate_requirement(req: str, product: dict[str, Any]) -> tuple[bool, str]:
    """Check if a specific requirement is satisfied by the product."""
    req_clean = req.strip().lower()
    searchable = " ".join([
        product["name"],
        product["brand"],
        product["description"],
        " ".join(f"{k}: {v}" for k, v in product["specifications"].items()),
        " ".join(product["features"]),
        " ".join(product["use_cases"]),
    ]).lower()

    budget_match = re.search(r"\b(?:under|below|less than|up to|max(?:imum)?)\s*(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\b", req_clean)
    if budget_match:
        val = float(budget_match.group(1).replace(",", ""))
        if budget_match.group(2):
            val *= 1000
        comp_price = product.get("price_inr") or product.get("price")
        if comp_price is not None and product["price_verified"]:
            if comp_price <= val:
                return True, f"Verified price ₹{int(comp_price):,} is within requirement of under ₹{int(val):,}"
            return False, f"Price ₹{int(comp_price):,} exceeds requirement of under ₹{int(val):,}"
        return False, "Price is unverified or unknown"

    if product["brand"] and product["brand"].lower() in req_clean:
        return True, f"Matches requested brand {product['brand']}"

    req_words = [w for w in re.findall(r"\b[a-zA-Z0-9-]+\b", req_clean) if len(w) > 2 and w not in {"for", "with", "and", "good", "best", "want", "need", "like"}]
    if not req_words:
        return True, "Matches general criteria"

    matched_words = [w for w in req_words if re.search(rf"\b{re.escape(w)}\b", searchable)]
    if len(matched_words) >= max(1, len(req_words) // 2):
        return True, f"Mentioned in specifications/features ({', '.join(matched_words)})"

    return False, "Not specified or verified in product information"


def compare_products_impl(
    products: list[dict[str, Any]],
    user_requirements: list[str] | None = None,
) -> dict[str, Any]:
    """
    Deterministically compares two or more products against available specs and user requirements.
    Never fabricates missing specifications or unverified claims.
    Ensures currency compatibility before direct price comparison.
    """
    if not products:
        return {
            "comparison": [],
            "common_features": [],
            "key_differences": [],
            "best_value": None,
            "best_match": None,
            "trade_offs": ["No products provided for comparison."],
            "missing_information": ["No product data supplied."],
            "summary": "No products provided for comparison.",
        }

    normalized = [_normalize_product_dict(p) for p in products]
    reqs = [r.strip() for r in (user_requirements or []) if r and r.strip()]

    # Currency consistency check
    currencies = {p["currency"] for p in normalized if p.get("currency")}
    has_multi_currency = len(currencies) > 1
    incompatible_currencies = any(not p["currency_compatible"] for p in normalized)

    # Infer relevant dimensions across the set
    all_dimensions: set[str] = set()
    combined_text = " ".join([p["name"] + " " + p["description"] for p in normalized]).lower()
    if any(k in combined_text for k in ("laptop", "notebook", "pc", "computer", "macbook")):
        all_dimensions.update(["Processor", "RAM / Memory", "Storage / SSD", "Display / Screen", "Battery / Runtime", "Operating System"])
    elif any(k in combined_text for k in ("headphone", "earphone", "earbud", "audio")):
        all_dimensions.update(["Noise Cancellation", "Battery / Runtime", "Warranty"])
    elif any(k in combined_text for k in ("luggage", "suitcase", "trolley", "bag", "backpack")):
        all_dimensions.update(["Capacity", "Weight", "Warranty"])
    else:
        all_dimensions.update(["Warranty", "Weight"])

    # Build per-product evaluation
    per_product: list[dict[str, Any]] = []
    prices_with_products: list[tuple[float, dict[str, Any]]] = []

    for p in normalized:
        available_specs, missing_specs = _extract_available_and_missing_specs(p, all_dimensions)

        matched_reqs: list[str] = []
        unmatched_reqs: list[str] = []
        for req in reqs:
            satisfied, reason = _evaluate_requirement(req, p)
            if satisfied:
                matched_reqs.append(f"{req} ({reason})")
            else:
                unmatched_reqs.append(f"{req} ({reason})")

        advantages: list[str] = []
        disadvantages: list[str] = []

        comp_price = p["price_inr"] if p["price_inr"] is not None else p["price"]
        if comp_price is not None and p["price_verified"] and not incompatible_currencies:
            prices_with_products.append((comp_price, p))

        if p["rating"] is not None and p["rating"] >= 4.0:
            advantages.append(f"Strong customer rating ({p['rating']}/5 from {p.get('review_count') or 0} reviews)")
        elif p["rating"] is not None and p["rating"] < 3.5:
            disadvantages.append(f"Lower customer rating ({p['rating']}/5)")

        if p["availability"] == "in_stock":
            advantages.append("In stock and available to purchase")
        elif p["availability"] == "out_of_stock":
            disadvantages.append("Currently marked out of stock")

        for m in matched_reqs:
            advantages.append(f"Fulfills requirement: {m}")

        per_product.append({
            "product_id": p["product_id"],
            "name": p["name"],
            "brand": p["brand"],
            "price": p["price"],
            "currency": p["currency"],
            "price_inr": p["price_inr"],
            "price_verified": p["price_verified"],
            "product_url": p["product_url"],
            "rating": p["rating"],
            "review_count": p["review_count"],
            "availability": p["availability"],
            "available_specifications": available_specs,
            "missing_specifications": missing_specs,
            "matched_requirements": matched_reqs,
            "unmatched_requirements": unmatched_reqs,
            "advantages": advantages,
            "disadvantages": disadvantages,
        })

    # Relative price advantages & disadvantages
    if len(prices_with_products) >= 2 and not incompatible_currencies:
        sorted_by_price = sorted(prices_with_products, key=lambda x: x[0])
        min_price, lowest_prod = sorted_by_price[0]
        max_price, highest_prod = sorted_by_price[-1]

        if max_price > min_price:
            for item in per_product:
                if item["product_id"] == lowest_prod["product_id"]:
                    item["advantages"].insert(0, f"Most affordable option in comparison set (₹{int(min_price):,})")
                elif item["product_id"] == highest_prod["product_id"]:
                    item["disadvantages"].insert(0, f"Highest price in comparison set (₹{int(max_price):,})")

    # Common features
    feature_counts: dict[str, int] = {}
    for p in normalized:
        for f in p["features"]:
            clean_f = f.strip()
            feature_counts[clean_f] = feature_counts.get(clean_f, 0) + 1

    common_features = [f for f, count in feature_counts.items() if count >= len(normalized) and len(f) > 3]

    # Key differences
    key_differences: list[str] = []
    if incompatible_currencies:
        key_differences.append(f"Incompatible currencies present ({', '.join(sorted(currencies))}); direct price ranking disabled")
    elif has_multi_currency:
        key_differences.append(f"Multiple currencies ({', '.join(sorted(currencies))}); normalized to benchmark INR for comparison")

    valid_prices = [p["price_inr"] for p in normalized if p["price_inr"] is not None and p["price_verified"]]
    if not incompatible_currencies and len(valid_prices) >= 2 and max(valid_prices) != min(valid_prices):
        key_differences.append(f"Price range: ₹{int(min(valid_prices)):,} to ₹{int(max(valid_prices)):,}")
    elif len(valid_prices) < len(normalized):
        key_differences.append("Price verification status varies across options")

    brands = {p["brand"] for p in normalized if p["brand"]}
    if len(brands) > 1:
        key_differences.append(f"Brands compared: {', '.join(sorted(brands))}")

    for dim in all_dimensions:
        dim_values = {}
        for p_eval in per_product:
            val = p_eval["available_specifications"].get(dim)
            if val:
                dim_values[p_eval["name"]] = val
        if len(dim_values) >= 2 and len(set(dim_values.values())) > 1:
            key_differences.append(f"{dim} differs across products")

    # Missing information aggregated
    missing_info: list[str] = []
    if incompatible_currencies:
        missing_info.append(f"Incompatible currencies ({', '.join(sorted(currencies))}) prevent direct price comparison.")

    for p_eval in per_product:
        if not p_eval["price_verified"] or p_eval["price"] is None:
            missing_info.append(f"{p_eval['name']}: Price is unverified")
        if p_eval["rating"] is None:
            missing_info.append(f"{p_eval['name']}: Rating unavailable (not treated as verified evidence)")
        for m in p_eval["missing_specifications"]:
            missing_info.append(f"{p_eval['name']}: {m}")

    # Determine best match & best value
    best_match = None
    best_value = None
    trade_offs: list[str] = []

    # Best match: highest number of matched requirements, then rating
    def match_sort_key(item: dict[str, Any]) -> tuple[int, float]:
        return (len(item["matched_requirements"]), item["rating"] or 0.0)

    ranked_by_match = sorted(per_product, key=match_sort_key, reverse=True)
    if ranked_by_match:
        top_m = ranked_by_match[0]
        match_count = len(top_m["matched_requirements"])
        best_match = {
            "product_id": top_m["product_id"],
            "name": top_m["name"],
            "reason": (
                f"Satisfies {match_count} of {len(reqs)} stated user requirements."
                if reqs else f"Highest specification and rating profile ({top_m.get('rating') or 'N/A'}/5)."
            ),
        }

    # Best value: evaluates verified requirement fulfillment relative to price
    valid_value_candidates = [
        item for item in per_product
        if item["price_verified"] and item.get("price_inr") is not None and item["price_inr"] > 0
    ]
    if valid_value_candidates and not incompatible_currencies:
        def value_score(item: dict[str, Any]) -> float:
            matches = len(item["matched_requirements"]) + 1
            # Unknown rating is neutral (1.0), not fabricated as 3.5
            rating_bonus = (item["rating"] / 5.0) if item["rating"] is not None else 1.0
            return (matches * rating_bonus) / item["price_inr"]

        sorted_by_val = sorted(valid_value_candidates, key=value_score, reverse=True)
        top_v = sorted_by_val[0]
        best_value = {
            "product_id": top_v["product_id"],
            "name": top_v["name"],
            "reason": f"Offers strong requirement coverage and features at a verified price of ₹{int(top_v['price_inr']):,}.",
        }
    else:
        best_value = None
        missing_info.insert(0, "Best value cannot be determined because verified prices are missing.")

    # Trade-offs synthesis
    if len(per_product) >= 2:
        p1 = per_product[0]
        p2 = per_product[1]
        p1_price_str = f"₹{int(p1['price_inr']):,}" if p1.get("price_inr") else "unverified price"
        p2_price_str = f"₹{int(p2['price_inr']):,}" if p2.get("price_inr") else "unverified price"

        if p1.get("price_inr") and p2.get("price_inr") and p1["price_inr"] != p2["price_inr"]:
            diff = abs(int(p1["price_inr"] - p2["price_inr"]))
            cheaper_p = p1 if p1["price_inr"] < p2["price_inr"] else p2
            costlier_p = p2 if p1["price_inr"] < p2["price_inr"] else p1
            trade_offs.append(
                f"{cheaper_p['name']} is ₹{diff:,} more affordable than {costlier_p['name']}, "
                f"while {costlier_p['name']} offers additional features or higher specifications."
            )
        else:
            trade_offs.append(f"{p1['name']} ({p1_price_str}) and {p2['name']} ({p2_price_str}) offer comparable price points with different feature focus.")

    # Build concise summary
    summary_parts = []
    if best_match:
        summary_parts.append(f"Best Match: {best_match['name']} ({best_match['reason']})")
    if best_value and (not best_match or best_value["product_id"] != best_match["product_id"]):
        summary_parts.append(f"Best Value: {best_value['name']} ({best_value['reason']})")
    elif best_value and best_match and best_value["product_id"] == best_match["product_id"]:
        summary_parts.append(f"Top Pick: {best_match['name']} is both the best match and best value.")

    return {
        "comparison": per_product,
        "common_features": common_features,
        "key_differences": key_differences,
        "best_value": best_value,
        "best_match": best_match,
        "trade_offs": trade_offs,
        "missing_information": missing_info[:10],
        "summary": " · ".join(summary_parts) if summary_parts else "Product comparison completed.",
    }
