import re

from backend.services.ai import gemini_json


def _as_list(value):
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [str(value).strip().lower()] if value else []


_CUR = r"(?:₹|rs\.?|inr|rupees?)?\s*"
_AMT = r"(\d[\d,]*(?:\.\d+)?)\s*(k)?\b"
_MAX_RE = re.compile(rf"\b(?:under|below|less than|up to|upto|within|at most|not more than|max(?:imum)?|budget(?: of)?)\s*{_CUR}{_AMT}")
_MIN_RE = re.compile(rf"\b(?:above|over|more than|greater than|at least|min(?:imum)?|starting (?:at|from))\s*{_CUR}{_AMT}")
_RANGE_RE = re.compile(rf"\bbetween\s*{_CUR}{_AMT}\s*(?:and|to|-)\s*{_CUR}{_AMT}")
_STOPWORDS = {"a","an","the","some","any","find","show","me","get","buy","i","want","need","looking","for","to","please","good","best","cheap","with","of","in","my","and","pair","under","above","suggest","gifts","gift","years","year","old","into","he","she","is"}
_VAGUE = {"gift","gifts","present","something","anything","stuff","thing","things","item","items","product","products","nice","sister","brother","dad","mom","father","mother","friend","wife","husband"}
_FOLLOWUP_RE = re.compile(r"\b(ones?|it|them|that|these|those|same|instead|also|only|cheaper|different|another|more|less|similar|make)\b")


def _amount(number: str, suffix: str | None) -> float:
    value = float(number.replace(",", ""))
    return value * 1000 if suffix else value


def extract_budget(lowered: str) -> tuple[float | None, float | None]:
    match = _RANGE_RE.search(lowered)
    if match:
        return _amount(match.group(1), match.group(2)), _amount(match.group(3), match.group(4))
    match = _MIN_RE.search(lowered)
    if match:
        return _amount(match.group(1), match.group(2)), None
    match = _MAX_RE.search(lowered)
    if match:
        return None, _amount(match.group(1), match.group(2))
    return None, None


def product_phrase(lowered: str) -> str:
    cleaned = _RANGE_RE.sub(" ", lowered)
    cleaned = _MIN_RE.sub(" ", cleaned)
    cleaned = _MAX_RE.sub(" ", cleaned)
    words = [word for word in re.findall(r"[a-z]+", cleaned) if word not in _STOPWORDS]
    if not words or all(word in _VAGUE for word in words):
        return ""
    return " ".join(words[:4])


def _canonical_category(value):
    text = str(value or "").lower()
    aliases = {
        "laptops": "laptops and electronics",
        "laptop": "laptops and electronics",
        "computer": "laptops and electronics",
        "computers": "laptops and electronics",
        "electronics": "laptops and electronics",
        "backpack": "backpacks",
        "rucksack": "backpacks",
        "skin care": "skincare",
        "skin-care": "skincare",
        "fitness": "fitness equipment",
        "gym": "fitness equipment",
        "hiking accessory": "hiking accessories",
        "outdoor equipment": "hiking accessories",
    }
    return aliases.get(text, text or None)


def _canonical_subcategory(value):
    text = str(value or "").lower()
    aliases = {
        "laptop": "Laptops",
        "laptops": "Laptops",
        "notebook computer": "Laptops",
        "wireless headphones": "Headphones",
        "running shoes": "Running",
        "office chair": "Office Chairs",
        "microwave": "Kitchen Appliances",
        "cabin suitcase": "Suitcases",
        "programming books": "Programming Books",
    }
    return aliases.get(text, value or None)


def _deterministic_intent(text: str) -> dict:
    lowered = text.lower()
    budget_min, budget_max = extract_budget(lowered)
    preferences = []
    preference_terms = {
        "tsa lock": ("tsa lock",),
        "lightweight": ("lightweight", "light weight"),
        "comfort": ("comfortable", "comfy", "comfort"),
        "joint-support": ("bad knees", "joint", "supportive"),
        "waterproof": ("waterproof", "water resistant"),
        "durable": ("durable", "long lasting", "sturdy"),
    }
    for preference, terms in preference_terms.items():
        if any(term in lowered for term in terms):
            preferences.append(preference)

    category = None
    category_terms = {
        "walking shoes": ("walking shoes", "shoes", "sneakers", "footwear"),
        "backpacks": ("backpack", "daypack", "rucksack"),
        "hiking accessories": (
            "trekking poles",
            "hiking poles",
            "headlamp",
            "hiking accessories",
            "outdoor equipment",
        ),
        "outdoor clothing": ("jacket", "socks", "clothing"),
        "outdoor safety": ("first aid", "safety"),
        "laptops and electronics": ("laptop", "computer", "electronics", "headphone", "earbuds", "earphones", "smartphone", "tablet"),
        "furniture and home decor": ("office chair", "desk chair", "bookshelf", "desk", "lamp", "furniture"),
        "travel and luggage": ("suitcase", "cabin bag", "luggage", "travel bag"),
        "home and kitchen": ("microwave", "air fryer", "coffee maker", "mixer", "cookware", "kitchen"),
        "fashion and clothing": ("t-shirt", "shirt", "jeans", "dress", "jacket", "sandals", "handbag"),
        "books and stationery": ("book", "notebook", "stationery", "pen", "programming book"),
        "grocery and household essentials": ("cereal", "coffee", "tea", "detergent", "cleaning supplies", "grocery"),
        "baby and kids": ("baby", "kids", "children", "toy", "school bag"),
        "automotive accessories": ("car", "vehicle", "tyre", "tire", "automotive"),
        "pet supplies": ("pet", "puppy", "dog", "cat", "leash"),
        "skincare": ("skincare", "skin care", "moisturizer", "serum", "cleanser"),
        "fitness equipment": ("fitness", "gym", "treadmill", "dumbbell", "exercise"),
    }
    for candidate, terms in category_terms.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
            category = candidate
            break
    if category is None and any(term in lowered for term in ("walking", "walks", "walk")) and any(
        term in lowered for term in ("comfortable", "comfy", "comfort", "feet")
    ):
        category = "walking shoes"

    activity = None
    activity_terms = {
        "hiking": ("hiking", "trekking", "trail"),
        "walking": ("walking", "walks", "walk"),
        "camping": ("camping", "camp"),
        "travel": ("travel", "travelling", "trip"),
    }
    for candidate, terms in activity_terms.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
            activity = candidate
            break

    subcategory = None
    subcategory_terms = {
        "Laptops": ("laptop", "notebook computer"),
        "Headphones": ("headphones", "earbuds", "earphones"),
        "Office Chairs": ("office chair", "desk chair"),
        "Kitchen Appliances": ("microwave", "air fryer", "coffee maker"),
        "Running": ("running shoes", "running sneaker"),
        "Suitcases": ("suitcase", "cabin bag", "cabin luggage"),
        "Programming Books": ("programming book", "coding book"),
    }
    for candidate, terms in subcategory_terms.items():
        if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms):
            subcategory = candidate
            break
    if category is None and subcategory in {"Laptops", "Headphones"}:
        category = "laptops and electronics"
    if category is None and subcategory in {"Office Chairs"}:
        category = "furniture and home decor"
    if category is None and subcategory in {"Suitcases"}:
        category = "travel and luggage"

    return {
        "category": category or ("general" if product_phrase(lowered) else None),
        "subcategory": subcategory,
        "budget_min": budget_min,
        "budget_max": budget_max,
        "product_type": product_phrase(lowered) or None,
        "recipient": "father" if any(term in lowered for term in ("dad", "father")) else None,
        "activity": activity,
        "preferences": preferences,
        "hard_constraints": ["in_stock"] if "in stock" in lowered or "available" in lowered else [],
    }


def extract_intent(state):
    text = state.get("normalized_query", state.get("message", ""))
    deterministic = _deterministic_intent(text)
    prompt = f"""Extract shopping intent from this customer request: {text!r}
Return JSON only with exactly these keys:
category (string or null), subcategory (string or null), product_type (string or null), budget_min (number or null), budget_max (number or null), recipient (string or null),
activity (string or null), preferences (array of short strings), hard_constraints (array), is_followup (boolean).
Do not invent a budget or product feature. Use null when unspecified."""
    llm_result = gemini_json(prompt)
    extracted = llm_result or deterministic
    extracted["category"] = _canonical_category(extracted.get("category"))
    previous_constraints = state.get("constraints", {})
    previous_category = _canonical_category(previous_constraints.get("category"))
    previous_subcategory = _canonical_subcategory(previous_constraints.get("subcategory"))
    is_short_followup = bool(_FOLLOWUP_RE.search(text.lower()) or not deterministic.get("product_type"))
    independent_terms = ("gift", "headphones", "earbuds", "laptop", "skincare", "office chair", "suitcase", "pet supplies")
    is_independent_query = any(term in text.lower() for term in independent_terms) and not any(
        phrase in text.lower() for phrase in ("make it", "only show", "same category", "different brand", "cheaper")
    )
    is_new_category = bool(
        previous_constraints
        and not is_short_followup
        and (
            (extracted.get("category") and extracted["category"] != previous_category)
            or (extracted.get("subcategory") and extracted.get("subcategory") != previous_subcategory)
            or (is_independent_query and not extracted.get("category"))
        )
    )
    constraints = {} if is_new_category else previous_constraints.copy()
    extracted["subcategory"] = _canonical_subcategory(
        extracted.get("subcategory") or _deterministic_intent(text).get("subcategory")
    )
    extracted["budget_min"] = deterministic.get("budget_min")
    extracted["budget_max"] = deterministic.get("budget_max")
    extracted["hard_constraints"] = deterministic.get("hard_constraints", [])
    extracted["subcategory"] = _canonical_subcategory(extracted.get("subcategory") or deterministic.get("subcategory"))
    for key in ("category", "subcategory", "product_type", "budget_min", "budget_max", "recipient", "activity"):
        if extracted.get(key) is not None:
            constraints[key] = extracted[key]
    preferences = list(dict.fromkeys(constraints.get("preferences", []) + _as_list(extracted.get("preferences"))))
    if preferences:
        constraints["preferences"] = preferences
    constraints["hard_constraints"] = list(dict.fromkeys(
        constraints.get("hard_constraints", []) + _as_list(extracted.get("hard_constraints"))
    ))
    return {
        "constraints": constraints,
        "intent_confidence": 0.9 if llm_result else 0.6,
        "intent_source": "gemini" if llm_result else "deterministic",
        "query_is_new": is_new_category,
    }