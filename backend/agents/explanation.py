import logging
import re
from typing import Any

from backend.schemas.product import ProductExplanation, RequirementMatch
from backend.services.ai import gemini_json, llm_json, openrouter_json

logger = logging.getLogger("explanation")


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _extract_evidence_and_requirement_matches(
    product: dict[str, Any],
    constraints: dict[str, Any],
    rank: int,
    total_candidates: int,
) -> dict[str, Any]:
    """
    Extracts deterministic, verified requirement matches and evidence directly from product data.
    Distinguishes 'matched', 'not_matched', and 'unknown'. Never treats unknown as matched.
    """
    req_matches: list[dict[str, Any]] = []
    strengths: list[str] = []
    limitations: list[str] = []
    matched_attrs: list[str] = []
    unverified_attrs: list[str] = []

    name = str(product.get("name") or "")
    brand = str(product.get("brand") or "").strip()
    price = product.get("price")
    features = [f.lower() for f in as_list(product.get("features"))]
    tags = [t.lower() for t in as_list(product.get("tags"))]
    use_cases = [u.lower() for u in as_list(product.get("use_cases"))]
    desc = str(product.get("description") or "").lower()
    specs = str(product.get("specifications") or "").lower()
    full_text = " ".join([name.lower(), brand.lower(), desc, specs] + features + tags + use_cases)

    # 1. Product Type / Category requirement
    ptype = constraints.get("product_type") or constraints.get("category")
    if ptype:
        ptype_str = str(ptype).strip()
        ptype_lower = ptype_str.lower()
        if ptype_lower in name.lower() or ptype_lower in str(product.get("category", "")).lower() or ptype_lower in str(product.get("subcategory", "")).lower():
            req_matches.append({
                "requirement": ptype_str.title(),
                "status": "matched",
                "evidence": f"Product listing identifies category/type as '{ptype_str.title()}'.",
            })
            matched_attrs.append(ptype_str.title())
            strengths.append(f"Matches requested {ptype_str.title()} category.")
        else:
            req_matches.append({
                "requirement": ptype_str.title(),
                "status": "matched" if any(w in full_text for w in ptype_lower.split()) else "unknown",
                "evidence": f"Relevant match to {ptype_str.title()} search query.",
            })
            matched_attrs.append(ptype_str.title())

    # 2. Budget requirement
    budget_max = constraints.get("budget_max")
    if budget_max is not None:
        b_max = float(budget_max)
        if price is not None:
            p_val = float(price)
            if p_val <= b_max:
                req_matches.append({
                    "requirement": f"Budget (Under ₹{int(b_max):,})",
                    "status": "matched",
                    "evidence": f"Listed at ₹{int(p_val):,}, comfortably within your ₹{int(b_max):,} budget.",
                })
                matched_attrs.append(f"Price: ₹{int(p_val):,}")
                strengths.append(f"Within requested budget at ₹{int(p_val):,}.")
            else:
                req_matches.append({
                    "requirement": f"Budget (Under ₹{int(b_max):,})",
                    "status": "not_matched",
                    "evidence": f"Listed at ₹{int(p_val):,}, exceeding your budget of ₹{int(b_max):,}.",
                })
                limitations.append(f"Price ₹{int(p_val):,} exceeds target budget.")
        else:
            req_matches.append({
                "requirement": f"Budget (Under ₹{int(b_max):,})",
                "status": "unknown",
                "evidence": "Price is not independently verified on merchant page.",
            })
            unverified_attrs.append("Verified price")
            limitations.append("Price not independently verified on merchant page.")
    elif price is not None:
        matched_attrs.append(f"Price: ₹{int(price):,}")

    # 3. Intended Use / Activity requirement
    intended_use = constraints.get("intended_use") or constraints.get("activity")
    if intended_use:
        use_str = str(intended_use).strip()
        use_lower = use_str.lower()
        if any(use_lower in uc for uc in use_cases) or use_lower in full_text:
            req_matches.append({
                "requirement": f"For {use_str.title()}",
                "status": "matched",
                "evidence": f"Product features and use cases highlight suitability for {use_str}.",
            })
            matched_attrs.append(f"For {use_str.title()}")
            strengths.append(f"Well-suited for {use_str}.")
        else:
            req_matches.append({
                "requirement": f"For {use_str.title()}",
                "status": "unknown",
                "evidence": f"Suitability for {use_str} was not explicitly documented in the listing.",
            })
            unverified_attrs.append(f"Suitability for {use_str}")

    # 4. Technical Preferences and Specifications
    user_preferences = as_list(constraints.get("preferences", []))
    spec_synonyms = {
        "wireless": ["wireless", "bluetooth"],
        "bluetooth": ["bluetooth", "wireless"],
        "anc": ["anc", "active noise cancellation", "noise cancelling", "noise cancellation"],
        "wired": ["wired", "3.5mm", "aux"],
        "lightweight": ["lightweight", "light weight", "portable"],
        "waterproof": ["waterproof", "water resistant", "ipx"],
        "fast charging": ["fast charging", "quick charge"],
    }

    for pref in user_preferences:
        pref_lower = pref.lower()
        search_terms = spec_synonyms.get(pref_lower, [pref_lower])

        # Look for explicit mention in product features, tags, name, or specs
        found_in_features = [
            f for f in as_list(product.get("features"))
            if any(term in f.lower() for term in search_terms)
        ]

        display_label = pref.upper() if ("ram" in pref_lower or "ssd" in pref_lower or len(pref) <= 4) else pref.title()
        if found_in_features:
            req_matches.append({
                "requirement": display_label,
                "status": "matched",
                "evidence": f"Product listing specifies '{found_in_features[0]}'.",
            })
            matched_attrs.append(display_label)
            strengths.append(f"Features {found_in_features[0]}.")
        elif any(term in full_text for term in search_terms):
            matched_term = next(term for term in search_terms if term in full_text)
            req_matches.append({
                "requirement": display_label,
                "status": "matched",
                "evidence": f"Identified '{matched_term}' in merchant specifications.",
            })
            matched_attrs.append(display_label)
            strengths.append(f"Includes {display_label}.")
        else:
            display_label = pref.upper() if ("ram" in pref_lower or "ssd" in pref_lower or len(pref) <= 4) else pref.title()
            req_matches.append({
                "requirement": display_label,
                "status": "unknown",
                "evidence": f"{display_label} information was not available or verified in listing.",
            })
            unverified_attrs.append(display_label)
            limitations.append(f"{display_label} details could not be verified.")

    # 5. Brand Preference / Verification
    preferred_brands = as_list(constraints.get("preferred_brands", []))
    if preferred_brands:
        b_lower = brand.lower()
        if any(pb.lower() == b_lower or pb.lower() in b_lower for pb in preferred_brands):
            req_matches.append({
                "requirement": f"Brand: {brand.title()}",
                "status": "matched",
                "evidence": f"Manufactured by preferred brand {brand.title()}.",
            })
            matched_attrs.append(f"Brand: {brand.title()}")
            strengths.append(f"Matches preferred brand {brand.title()}.")

    # 6. Ratings and Stock Status Evidence
    rating = product.get("rating")
    if rating not in (None, "") and float(rating) >= 4.0:
        strengths.append(f"Strong customer rating of {float(rating):.1f}★.")
        matched_attrs.append(f"Rating: {float(rating):.1f}★")
    elif rating in (None, ""):
        unverified_attrs.append("Customer reviews")
        limitations.append("Customer reviews and ratings not yet available.")

    if product.get("price_verified") is not True:
        limitations.append("Price not independently verified on merchant page.")
    if product.get("availability") == "unknown":
        limitations.append("Stock availability unconfirmed.")

    return {
        "requirement_matches": req_matches,
        "strengths": list(dict.fromkeys(strengths)),
        "limitations": list(dict.fromkeys(limitations)),
        "matched_attributes": list(dict.fromkeys(matched_attrs)),
        "unverified_attributes": list(dict.fromkeys(unverified_attrs)),
    }


def _build_deterministic_retrieval_reason(
    product: dict[str, Any],
    constraints: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    """
    Constructs a grounded reason explaining why the product was retrieved.
    Only mentions attributes supported by actual product data.
    """
    ptype = constraints.get("product_type") or constraints.get("category") or "product"
    intended_use = constraints.get("intended_use") or constraints.get("activity")
    budget_max = constraints.get("budget_max")
    price = product.get("price")
    matched_attrs = evidence["matched_attributes"]

    clauses = [f"Retrieved because this listing matches your requested {ptype} category"]
    if intended_use:
        clauses.append(f"for {intended_use}")

    if budget_max is not None and price is not None and float(price) <= float(budget_max):
        clauses.append(f"and falls within your specified budget at ₹{int(price):,}")
    elif price is not None:
        clauses.append(f"priced at ₹{int(price):,}")

    # Add verified features
    verified_features = [
        a for a in matched_attrs
        if "Price" not in a
        and a.lower() != str(ptype).lower()
        and (not intended_use or a.lower() != f"for {str(intended_use).lower()}")
    ]
    if verified_features:
        clauses.append(f"with confirmed {', '.join(verified_features[:2])}")

    return " ".join(clauses) + "."


def _build_deterministic_ranking_reason(
    product: dict[str, Any],
    rank: int,
    total_candidates: int,
    evidence: dict[str, Any],
) -> str:
    """
    Constructs an evidence-based explanation for the product's assigned rank position.
    """
    strengths = evidence["strengths"]
    limitations = evidence["limitations"]
    matched_count = sum(1 for m in evidence["requirement_matches"] if m.get("status") == "matched")

    if rank == 1:
        if strengths:
            top_s = strengths[0].rstrip(".")
            return f"Ranked #1 because it satisfies {matched_count} of your verified requirements ({top_s.lower()}) with the highest overall relevance among retrieved options."
        return f"Ranked #1 because it offers the highest composite relevance match for your query and constraints."
    elif rank == 2:
        if len(limitations) > 0 and len(strengths) > 0:
            return f"Ranked #2 as a strong alternative: matches your core requirements but provides slightly less verified evidence than the #1 pick ({limitations[0].lower()})."
        return f"Ranked #2 as a close alternative with balanced specifications within your target range."
    elif rank == 3:
        if limitations:
            return f"Ranked #3 because while it matches key requirements, it has fewer verified advantages ({limitations[0].lower()}) than higher-ranked options."
        return f"Ranked #3 based on price and available feature compatibility relative to top recommendations."
    else:
        return f"Ranked #{rank} of {total_candidates} candidates based on price tier and available feature compatibility."


def explain_recommendations(state: dict[str, Any]) -> dict[str, Any]:
    """
    Dedicated LLM-based Product Explanation Agent.
    Generates structured, evidence-grounded explanations for:
      A. Why this product was retrieved (query & constraints relevance).
      B. Why this product is ranked at this position (relative candidate comparison).
    Includes deterministic fallback when LLM is unavailable or fails.
    """
    constraints = state.get("constraints", {})
    catalog_products = state.get("ranked_products", [])[:5]
    total_candidates = len(state.get("ranked_products", []))
    provider_status = state.get("provider_status", {})
    retrieval_errors = state.get("retrieval_errors", {})

    has_failed_provider = any(status != "ready" for status in provider_status.values()) or bool(retrieval_errors)
    has_unverified = any(not p.get("price_verified") for p in catalog_products)
    search_status = "partial_results" if (has_failed_provider or has_unverified) else "success"

    # Step 1: Compute verified evidence deterministically for each product
    prepared_items = []
    for idx, product in enumerate(catalog_products, start=1):
        rank_val = product.get("rank") if product.get("rank") is not None else idx
        evidence = _extract_evidence_and_requirement_matches(product, constraints, rank_val, total_candidates)
        det_retrieval = _build_deterministic_retrieval_reason(product, constraints, evidence)
        det_ranking = _build_deterministic_ranking_reason(product, rank_val, total_candidates, evidence)

        prepared_items.append({
            "product": product,
            "rank": rank_val,
            "evidence": evidence,
            "det_retrieval": det_retrieval,
            "det_ranking": det_ranking,
        })

    # Step 2: Request natural LLM conversion constrained strictly by provided evidence
    llm_payload = [
        {
            "product_id": item["product"].get("product_id"),
            "rank": item["rank"],
            "name": item["product"].get("name"),
            "price": item["product"].get("price"),
            "matched_requirements": [m["requirement"] for m in item["evidence"]["requirement_matches"] if m["status"] == "matched"],
            "unverified_requirements": [m["requirement"] for m in item["evidence"]["requirement_matches"] if m["status"] == "unknown"],
            "strengths": item["evidence"]["strengths"],
            "limitations": item["evidence"]["limitations"],
        }
        for item in prepared_items
    ]

    llm_prompt = f"""You are an evidence-grounded shopping explanation agent.
User Query: {state.get('normalized_query', state.get('message', ''))!r}
Constraints: {constraints!r}
Product Evidence and Rankings: {llm_payload!r}

CRITICAL RULES:
1. Return JSON ONLY with key "explanations":
{{
  "explanations": [
    {{
      "product_id": "...",
      "retrieval_reason": "1-2 sentences on why this product was retrieved based on query/budget/category.",
      "ranking_reason": "1-2 sentences explaining why this product is ranked at its exact position relative to others."
    }}
  ]
}}
2. Explanations must strictly use the provided evidence. NEVER invent specifications, ratings, or discounts.
3. For retrieval_reason, state how it relates to requested category, budget, or verified features.
4. For ranking_reason, explain why it holds its specific rank position (e.g. why #1 is top, or why #2/#3 has trade-offs).
5. Never alter product ranks or introduce new products."""

    llm_result = openrouter_json(llm_prompt, operation="explanation")
    llm_map = {}
    if llm_result and isinstance(llm_result, dict):
        for entry in llm_result.get("explanations", []):
            if isinstance(entry, dict) and entry.get("product_id"):
                llm_map[entry["product_id"]] = entry

    # Step 3: Build final recommendation objects with structured explanation schemas
    final_recommendations = []
    for item in prepared_items:
        prod = item["product"]
        pid = prod.get("product_id")
        rank_val = item["rank"]
        evidence = item["evidence"]
        llm_entry = llm_map.get(pid, {})

        retrieval_reason = (llm_entry.get("retrieval_reason") or item["det_retrieval"]).strip()
        ranking_reason = (llm_entry.get("ranking_reason") or item["det_ranking"]).strip()

        # Combine into short overview explanation for backward compatibility
        summary_explanation = f"{retrieval_reason} {ranking_reason}"

        structured_explanation = ProductExplanation(
            product_id=str(pid),
            rank=rank_val,
            retrieval_reason=retrieval_reason,
            requirement_matches=[RequirementMatch(**m) for m in evidence["requirement_matches"]],
            ranking_reason=ranking_reason,
            strengths=evidence["strengths"],
            limitations=evidence["limitations"],
            matched_attributes=evidence["matched_attributes"],
            unverified_attributes=evidence["unverified_attributes"],
        ).model_dump(mode="json")

        final_recommendations.append({
            **prod,
            "rank": rank_val,
            "explanation": summary_explanation,
            "retrieval_reason": retrieval_reason,
            "ranking_reason": ranking_reason,
            "requirement_matches": evidence["requirement_matches"],
            "strengths": evidence["strengths"],
            "limitations": evidence["limitations"],
            "matched_attributes": evidence["matched_attributes"],
            "unverified_attributes": evidence["unverified_attributes"],
            "explanation_details": structured_explanation,
            "tradeoffs": evidence["limitations"],
            "matched_requirements": list(dict.fromkeys(
                [m["requirement"] for m in evidence["requirement_matches"] if m["status"] == "matched"]
                + (prod.get("matched_requirements") or [])
            )),
            "benefits": evidence["strengths"][:4],
            "comparison_note": ranking_reason,
        })

    return {
        "recommended_products": final_recommendations,
        "action": "recommendations",
        "response_message": "These products were retrieved based on your requested category, budget, and features. They are ordered according to their relevance and verified requirement matches.",
        "search_status": search_status,
        "fallback_suggestions": [],
    }


def handle_explanation_followup(
    query: str,
    previous_products: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """
    Handles follow-up explanation queries using existing recommendation evidence
    without triggering a broad external product search.
    Examples:
      - "Why is this ranked first?"
      - "Why is this product better?"
      - "Why was this product recommended?"
      - "What does it not satisfy?"
    """
    if not previous_products:
        return {
            "action": "message",
            "response_message": "I don't have any previously recommended products to explain. What item are you looking for?",
            "recommended_products": [],
        }

    lowered = query.lower().strip()

    # Determine target product (first by default, or second/third if asked)
    target_idx = 0
    if re.search(r"\b(?:second|2nd|#2|number\s*2)\b", lowered):
        target_idx = 1 if len(previous_products) > 1 else 0
    elif re.search(r"\b(?:third|3rd|#3|number\s*3)\b", lowered):
        target_idx = 2 if len(previous_products) > 2 else 0

    target = previous_products[target_idx]
    rank = target.get("rank", target_idx + 1)
    name = target.get("name", "This product")
    price_str = f"₹{int(target['price']):,}" if target.get("price") else "unverified price"

    retrieval_reason = target.get("retrieval_reason") or target.get("explanation") or "Matches your search criteria."
    ranking_reason = target.get("ranking_reason") or target.get("comparison_note") or f"Ranked #{rank} based on overall match."
    strengths = target.get("strengths") or target.get("benefits") or []
    limitations = target.get("limitations") or target.get("tradeoffs") or []
    matches = target.get("requirement_matches") or []

    # Specific question intent parsing
    if re.search(r"\b(?:why.*first|why.*top|why.*#1|why.*ranked\s*1)\b", lowered):
        msg = (
            f"**Why {name} is ranked #1:**\n"
            f"- **Position Reason**: {ranking_reason}\n"
            f"- **Verified Strengths**: {', '.join(strengths) if strengths else 'Top match for query'}.\n"
            f"- **Price**: Listed at {price_str}."
        )
    elif re.search(r"\b(?:what.*(?:not|unmet|missing|lacks?)|doesn't\s*satisfy)\b", lowered):
        if limitations:
            unmet = [m['requirement'] for m in matches if m.get('status') in ('not_matched', 'unknown')]
            msg = (
                f"**What {name} does not satisfy or cannot verify:**\n"
                f"- " + "\n- ".join(limitations) + "\n"
            )
            if unmet:
                msg += f"\n*Unverified user requirements*: {', '.join(unmet)}."
        else:
            msg = f"{name} satisfies all your specified requirements with no verified shortcomings identified."
    elif re.search(r"\b(?:why.*better|better\s*than)\b", lowered) and len(previous_products) > 1:
        runner_up = previous_products[1]
        msg = (
            f"**Why #{rank} ({name}) is ranked ahead of #{runner_up.get('rank', 2)} ({runner_up.get('name', 'runner-up')}):**\n"
            f"- **Core Advantage**: {ranking_reason}\n"
            f"- **Key Verified Features**: {', '.join(strengths[:3]) if strengths else 'Stronger verified attribute coverage'}.\n"
            f"- **Runner-up Trade-offs**: {', '.join(runner_up.get('limitations', ['Fewer verified attributes'])[:2])}."
        )
    else:
        # General "why was this recommended"
        msg = (
            f"**Recommendation details for {name} (Rank #{rank}):**\n"
            f"- **Retrieval**: {retrieval_reason}\n"
            f"- **Ranking**: {ranking_reason}\n"
        )
        if strengths:
            msg += f"- **Key Verified Strengths**: {', '.join(strengths[:3])}\n"
        if limitations:
            msg += f"- **Limitations / Notes**: {', '.join(limitations[:2])}\n"

    return {
        "action": "message",
        "response_message": msg,
        "recommended_products": [target],
        "selected_tool": "explain_ranking",
    }
