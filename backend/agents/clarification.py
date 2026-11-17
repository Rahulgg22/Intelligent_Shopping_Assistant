import re
from typing import Any

from backend.services.ai import gemini_json, llm_json, openrouter_json

_BROAD_CATEGORY_TERMS = {
    "travel equipment", "travel gear", "travel accessories", "travel items", "travel stuff",
    "electronics", "electronic products", "electronic items", "electronic goods", "gadgets", "gadget",
    "appliances", "home appliances", "kitchen appliances",
    "sports equipment", "sports gear", "fitness equipment", "fitness gear", "gym equipment", "gym gear", "workout gear",
    "furniture", "home decor", "home furnishing",
    "clothing", "clothes", "apparel", "wear",
    "footwear",
    "skincare", "skincare products", "beauty products", "makeup", "cosmetics",
    "books", "stationery",
    "gifts", "presents",
}

_BROAD_GROUP_SUFFIXES = re.compile(
    r"\b([a-z]+)\s+(equipment|gear|accessories|supplies|items|products|goods|devices|stuff)\b",
    re.I
)


def evaluate_clarification_decision(constraints: dict[str, Any], query: str = "") -> tuple[str, str | None]:
    """
    Evaluates whether shopping intent is sufficiently complete to perform product retrieval,
    or whether clarification is required.

    Distinguishes:
        - Broad shopping categories (require clarification even if category is recognized)
        - Specific product types (evaluate completeness of use, specs, size, budget)
        - Gift requests (require recipient + interests/budget)
        - Flexible budget instructions ('any budget', 'surprise me', 'show me anything')

    Returns:
        (decision, missing_detail)
        decision is one of:
            - "ASK_CLARIFICATION"
            - "SEARCH_PRODUCTS"
            - "UPDATE_EXISTING_PREFERENCES"
            - "HANDLE_FOLLOWUP"
    """
    product_type = constraints.get("product_type")
    category = constraints.get("category")
    recipient = constraints.get("recipient")
    budget_max = constraints.get("budget_max")
    budget_min = constraints.get("budget_min")
    flexible_budget = constraints.get("flexible_budget", False)
    intended_use = constraints.get("intended_use") or constraints.get("activity")
    preferences = constraints.get("preferences", [])
    size = constraints.get("size")
    preferred_brands = constraints.get("preferred_brands", [])
    is_followup = constraints.get("is_followup", False)
    brand_diversity = constraints.get("brand_diversity", False)
    cheaper = constraints.get("cheaper", False)
    is_broad_category = constraints.get("is_broad_category", False)

    lowered = query.lower().strip()
    pt_lower = str(product_type or "").lower().strip()

    # 1. Follow-up handling
    if is_followup:
        has_new_detail = (
            budget_max is not None
            or budget_min is not None
            or flexible_budget
            or brand_diversity
            or cheaper
            or bool(preferences)
            or bool(size)
            or bool(intended_use)
            or bool(preferred_brands)
        )
        if has_new_detail and (product_type or category or recipient):
            # If user provided budget or flexible budget, we are ready to search
            if budget_max is not None or budget_min is not None or flexible_budget:
                return "UPDATE_EXISTING_PREFERENCES", None
            # If user provided a concrete product type that is not broad
            if product_type and pt_lower not in _BROAD_CATEGORY_TERMS and not _BROAD_GROUP_SUFFIXES.search(pt_lower):
                return "UPDATE_EXISTING_PREFERENCES", None
            # If user provided intended use or specs
            if intended_use or preferences or size:
                return "UPDATE_EXISTING_PREFERENCES", None

        if any(term in lowered for term in ("cheaper", "different brand", "other brand", "similar", "options", "more", "relax", "increase budget")):
            return "HANDLE_FOLLOWUP", None

    # 2. Completely missing product type and category
    if not product_type and not category:
        if intended_use:
            return "ASK_CLARIFICATION", f"product_type_for_{intended_use}"
        return "ASK_CLARIFICATION", "product_category"

    # 3. Gift queries
    if category == "Gifts" or pt_lower in ("gift", "gifts", "present", "presents"):
        has_budget = budget_max is not None or budget_min is not None or flexible_budget
        if has_budget:
            return "SEARCH_PRODUCTS", None
        if recipient:
            return "ASK_CLARIFICATION", f"gift_for_{recipient}"
        return "ASK_CLARIFICATION", "gift_recipient_and_budget"

    # 4. Broad Shopping Category (Category recognized, but request is too broad)
    # A recognized category alone must not automatically mean the request is ready for retrieval.
    is_broad = (
        is_broad_category
        or pt_lower in _BROAD_CATEGORY_TERMS
        or bool(_BROAD_GROUP_SUFFIXES.search(pt_lower))
        or (category and not product_type)
    )

    if is_broad:
        # Check specific broad domains
        if any(k in pt_lower for k in ("travel equipment", "travel gear", "travel accessories", "travel stuff", "travel items")) or (category == "Travel and Luggage" and not product_type):
            return "ASK_CLARIFICATION", "travel_equipment"
        if any(k in pt_lower for k in ("electronic", "electronics", "gadget", "gadgets")) or (category == "Electronics" and not product_type):
            return "ASK_CLARIFICATION", "electronics"
        if any(k in pt_lower for k in ("sport", "fitness", "gym", "workout")):
            return "ASK_CLARIFICATION", "sports_fitness"
        if any(k in pt_lower for k in ("furniture", "decor", "home decor")):
            return "ASK_CLARIFICATION", "furniture"
        if any(k in pt_lower for k in ("skincare", "beauty", "cosmetic")):
            return "ASK_CLARIFICATION", "skincare"
        if any(k in pt_lower for k in ("appliance", "appliances", "kitchen appliance")):
            return "ASK_CLARIFICATION", "appliances"
        if any(k in pt_lower for k in ("cloth", "apparel", "clothing", "wear")):
            return "ASK_CLARIFICATION", "clothing"
        if "footwear" in pt_lower:
            return "ASK_CLARIFICATION", "footwear"
        if "book" in pt_lower:
            return "ASK_CLARIFICATION", "books"
        return "ASK_CLARIFICATION", f"broad_category_{pt_lower or category}"

    # 5. Specific Product Types - Completeness Evaluation
    has_budget = budget_max is not None or budget_min is not None or flexible_budget
    has_use = bool(intended_use)
    has_specs = len(preferences) > 0 or bool(size) or len(preferred_brands) > 0
    has_size = bool(size) or any(s in pt_lower for s in ("cabin", "carry-on", "check-in", "40l", "50l", "60l"))

    # If the user gave an explicit budget (or flexible budget instruction), request is bounded
    if has_budget:
        return "SEARCH_PRODUCTS", None

    # Suitcase / Luggage specific rule:
    # "Suggest a suitcase for international travel" -> lacks both budget and size/type
    if any(k in pt_lower for k in ("suitcase", "luggage", "travel bag", "trolley")):
        if not has_size and not has_budget:
            return "ASK_CLARIFICATION", "suitcase"
        if not has_use and not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "suitcase"

    # Backpack specific rule:
    # "Suggest a backpack" -> lacks use, specs, and budget
    if "backpack" in pt_lower:
        if not has_use and not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "backpack"

    # Headphones specific rule:
    # "Suggest headphones" -> lacks primary use and budget
    if any(k in pt_lower for k in ("headphone", "earbud", "earphone")):
        if not has_use and not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "headphones"

    # Laptop specific rule:
    # "Suggest a laptop" -> lacks budget and use
    if any(k in pt_lower for k in ("laptop", "computer", "notebook")):
        if not has_use and not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "laptop"

    # Shoes specific rule:
    # "Suggest shoes" -> lacks style/use and budget
    if any(k in pt_lower for k in ("shoe", "sneaker", "footwear")):
        if not has_use and not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "shoes"

    # Mobile phone / Smartphone:
    if any(k in pt_lower for k in ("phone", "mobile", "smartphone")):
        if not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "mobile_phone"

    # Washing machine:
    if "washing machine" in pt_lower:
        if not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "washing_machine"

    # Smartwatch:
    if "smartwatch" in pt_lower:
        if not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "smartwatch"

    # Furniture:
    if "furniture" in pt_lower:
        if not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "furniture"

    # Skincare:
    if "skincare" in pt_lower or "beauty" in pt_lower:
        if not has_specs and not has_budget:
            return "ASK_CLARIFICATION", "skincare"

    # Books:
    if "book" in pt_lower:
        if not has_specs and not has_use and not has_budget:
            return "ASK_CLARIFICATION", "books"

    # If already sufficiently specific, proceed to retrieval
    return "SEARCH_PRODUCTS", None


def _fallback_question(constraints: dict[str, Any], query: str = "", missing_detail: str | None = None) -> str:
    """
    Contextual, natural clarification generator.
    Tailors questions precisely to what is missing.
    """
    recipient = constraints.get("recipient")
    category = constraints.get("category")
    product_type = constraints.get("product_type")
    intended_use = constraints.get("intended_use") or constraints.get("activity")
    detail = str(missing_detail or "").lower()

    # 1. Travel Equipment / Travel Gear
    if "travel_equipment" in detail or any(k in str(product_type).lower() for k in ("travel equipment", "travel gear", "travel accessories")):
        return "What kind of travel are you preparing for: holidays, backpacking, business travel or camping, and what items do you need?"

    # 2. Electronics / Gadgets
    if "electronics" in detail or str(product_type).lower() in ("electronics", "electronic products", "electronic items", "gadgets"):
        return "Which type of electronic product are you looking for (such as a laptop, smartphone, or headphones)?"

    # 3. Sports & Fitness Equipment
    if "sports_fitness" in detail or any(k in str(product_type).lower() for k in ("sports equipment", "fitness equipment", "gym equipment", "workout gear")):
        return "What sport or fitness activity are you shopping for, and what equipment do you need?"

    # 4. Furniture & Home Decor
    if "furniture" in detail or "furniture" in str(product_type).lower():
        return "What type of furniture are you looking for (such as an office chair, desk, or sofa), and what is your approximate budget?"

    # 5. Skincare & Beauty
    if "skincare" in detail or any(k in str(product_type).lower() for k in ("skincare", "beauty")):
        return "What type of skincare product are you looking for (such as a moisturizer, serum, or cleanser), and what is your skin type?"

    # 6. Appliances
    if "appliance" in detail or "appliance" in str(product_type).lower():
        return "Which kitchen or home appliance are you looking for (such as an air fryer, microwave, or washing machine), and what is your budget?"

    # 7. Clothing & Apparel
    if "clothing" in detail or any(k in str(product_type).lower() for k in ("clothing", "clothes", "apparel")):
        return "What type of clothing are you looking for, and do you have a preferred style or occasion?"

    # 8. Books
    if "book" in detail or "book" in str(product_type).lower():
        return "What genres or topics are you interested in (such as programming, fiction, or science)?"

    # 9. Gifts
    if category == "Gifts" or str(product_type).lower() in ("gift", "gifts") or "gift" in detail:
        if recipient:
            return f"What are their interests or hobbies, and how much would you like to spend on the gift for your {recipient}?"
        return "Who is the gift for, and what are their interests or your approximate budget?"

    # 10. Suitcase / Luggage
    if "suitcase" in detail or any(k in str(product_type).lower() for k in ("suitcase", "luggage", "travel bag")):
        return "What is your approximate budget, or what size suitcase (such as a cabin bag or check-in luggage) are you looking for?"

    # 11. Backpack
    if "backpack" in detail or "backpack" in str(product_type).lower():
        return "What will you primarily use the backpack for (such as college, travel, or trekking), and do you have an approximate budget?"

    # 12. Headphones
    if "headphone" in detail or any(k in str(product_type).lower() for k in ("headphone", "earbud", "earphone")):
        return "What will you primarily use the headphones for, and what's your approximate budget?"

    # 13. Laptop / Computer
    if "laptop" in detail or any(k in str(product_type).lower() for k in ("laptop", "computer")):
        return "What is your approximate budget, or what will you primarily use the laptop for (such as programming, gaming, or daily work)?"

    # 14. Shoes / Footwear
    if "shoe" in detail or any(k in str(product_type).lower() for k in ("shoe", "sneaker", "footwear")):
        return "Are you looking for running, casual, sports, or formal shoes, and what is your approximate budget?"

    # 15. Mobile phone / Smartphone
    if "phone" in detail or any(k in str(product_type).lower() for k in ("phone", "mobile")):
        return "What is your approximate budget, and do you have any brand or camera preferences?"

    # 16. Smartwatch
    if "smartwatch" in detail or "smartwatch" in str(product_type).lower():
        return "What is your approximate budget, and are you looking for a fitness smartwatch or a daily lifestyle watch?"

    # 17. Washing machine
    if "washing_machine" in detail or "washing machine" in str(product_type).lower():
        return "What is your approximate budget, and do you prefer a front-load or top-load washing machine?"

    # 18. Generic / missing product type
    if not product_type:
        if intended_use:
            return f"What type of product are you looking for for {intended_use} (such as luggage, gear, or clothing)?"
        return "What type of product are you looking for, and do you have an approximate budget in mind?"

    return f"What is your approximate budget for the {product_type}, or do you have any specific preferences?"


def check_clarification(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Evaluates whether clarification is required before retrieval.
    Decides between ASK_CLARIFICATION, SEARCH_PRODUCTS, UPDATE_EXISTING_PREFERENCES, HANDLE_FOLLOWUP.
    """
    tool_intent = state.get("tool_intent")
    previous_products = state.get("previous_products") or []

    if tool_intent == "compare":
        if len(previous_products) < 2:
            return {
                "needs_clarification": True,
                "clarification_decision": "ASK_CLARIFICATION",
                "clarifying_question": "Please search for products first or specify at least two products you'd like to compare.",
                "pending_clarification": "compare_products",
            }
        return {
            "needs_clarification": False,
            "clarification_decision": "TOOL_COMPARE",
            "clarifying_question": None,
            "pending_clarification": None,
        }

    if tool_intent == "reviews":
        if not previous_products:
            return {
                "needs_clarification": True,
                "clarification_decision": "ASK_CLARIFICATION",
                "clarifying_question": "Which product would you like to see reviews for?",
                "pending_clarification": "product_reviews",
            }
        return {
            "needs_clarification": False,
            "clarification_decision": "TOOL_REVIEWS",
            "clarifying_question": None,
            "pending_clarification": None,
        }

    if tool_intent == "alternatives":
        if not previous_products:
            return {
                "needs_clarification": True,
                "clarification_decision": "ASK_CLARIFICATION",
                "clarifying_question": "Which product would you like to find alternatives for?",
                "pending_clarification": "product_alternatives",
            }
        return {
            "needs_clarification": False,
            "clarification_decision": "TOOL_ALTERNATIVES",
            "clarifying_question": None,
            "pending_clarification": None,
        }

    if tool_intent == "explain_ranking":
        if not previous_products:
            return {
                "needs_clarification": True,
                "clarification_decision": "ASK_CLARIFICATION",
                "clarifying_question": "Which recommendation would you like me to explain?",
                "pending_clarification": "product_explanation",
            }
        return {
            "needs_clarification": False,
            "clarification_decision": "TOOL_EXPLAIN",
            "clarifying_question": None,
            "pending_clarification": None,
        }

    constraints = dict(state.get("constraints", {}))
    query = state.get("normalized_query", state.get("message", ""))

    decision, missing_detail = evaluate_clarification_decision(constraints, query)

    if decision != "ASK_CLARIFICATION":
        return {
            "needs_clarification": False,
            "clarification_decision": decision,
            "clarifying_question": None,
            "pending_clarification": None,
        }

    # Generate a dynamic question: try Gemini LLM first
    llm_question = None
    recipient = constraints.get("recipient")
    recipient_instruction = (
        f"The recipient is explicitly {recipient}. Mention {recipient} or the appropriate pronoun. NEVER mention sister or any other recipient."
        if recipient
        else "Do not assume or invent a recipient."
    )

    prompt = f"""You are a helpful, concise shopping assistant.
Customer query: {query!r}
Known shopping constraints: {constraints!r}
Missing information needed: {missing_detail}

Generate ONE natural, direct clarification question for the customer.
CRITICAL RULES:
1. {recipient_instruction}
2. Never assume outdoor or hiking equipment unless the user explicitly requested it.
3. Keep the question friendly, conversational, concise, and focused on: {missing_detail}.
4. Return JSON only: {{"question": "..."}}"""

    llm_response = openrouter_json(prompt, operation="clarification")
    if llm_response and isinstance(llm_response, dict) and llm_response.get("question"):
        llm_question = str(llm_response["question"]).strip()

    question = llm_question or _fallback_question(constraints, query, missing_detail)

    # Avoid repeatedly asking the exact same question if the user already answered
    last_question = state.get("pending_clarification_question") or constraints.get("last_clarifying_question")
    if last_question and question.strip().lower() == last_question.strip().lower():
        return {
            "needs_clarification": False,
            "clarification_decision": "SEARCH_PRODUCTS",
            "clarifying_question": None,
            "pending_clarification": None,
        }

    constraints["last_clarifying_question"] = question

    return {
        "needs_clarification": True,
        "clarification_decision": "ASK_CLARIFICATION",
        "clarifying_question": question,
        "pending_clarification": missing_detail,
        "pending_clarification_question": question,
        "constraints": constraints,
    }
