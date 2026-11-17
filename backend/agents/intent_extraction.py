import re
from typing import Any

from backend.services.ai import gemini_json, llm_json, openrouter_json


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        if ";" in value:
            return [item.strip().lower() for item in value.split(";") if item.strip()]
        if "," in value:
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return [value.strip().lower()]
    return []


_CUR = r"(?:₹|rs\.?|inr|rupees?)?\s*"
_AMT = r"(\d[\d,]*(?:\.\d+)?)\s*(k)?\b"
_MAX_RE = re.compile(rf"\b(?:under|below|less than|up to|upto|within|at most|not more than|max(?:imum)?|budget(?: of)?)\s*{_CUR}{_AMT}", re.I)
_MIN_RE = re.compile(rf"\b(?:above|over|more than|greater than|at least|min(?:imum)?|starting (?:at|from))\s*{_CUR}{_AMT}", re.I)
_RANGE_RE = re.compile(rf"\b(?:between\s*)?{_CUR}{_AMT}\s*(?:to|-|and)\s*{_CUR}{_AMT}\b", re.I)
_BARE_BUDGET_RE = re.compile(rf"^\s*{_CUR}{_AMT}\s*$", re.I)

_COMMAND_PHRASES = [
    r"\bsuggest\s+me\b", r"\bsuggest\s+a\b", r"\bsuggest\b",
    r"\brecommend\s+me\b", r"\brecommend\s+a\b", r"\brecommend\b",
    r"\bfind\s+me\b", r"\bfind\s+a\b", r"\bfind\b",
    r"\bshow\s+me\b", r"\bshow\b",
    r"\blooking\s+for\b", r"\blook\s+for\b",
    r"\bi\s+want\s+to\s+buy\b", r"\bi\s+want\b",
    r"\bi\s+need\s+a\b", r"\bi\s+need\b",
    r"\bget\s+me\b", r"\bget\s+a\b", r"\bget\b",
    r"\bbuy\s+a\b", r"\bbuy\b",
    r"\bhelp\s+me\s+find\b",
]

_RECIPIENT_PATTERNS = {
    "brother": [r"\bbrothers?\b", r"\bbros?\b"],
    "sister": [r"\bsisters?\b", r"\bsis\b"],
    "father": [r"\bfathers?\b", r"\bdads?\b", r"\bdaddy\b", r"\bpapa\b"],
    "mother": [r"\bmothers?\b", r"\bmoms?\b", r"\bmommy\b", r"\bmumma\b", r"\bma\b"],
    "friend": [r"\bfriends?\b", r"\bbestie\b", r"\bbuddy\b", r"\bpal\b"],
    "wife": [r"\bwifes?\b", r"\bwives\b"],
    "husband": [r"\bhusb(?:and)?s?\b", r"\bhubby\b"],
    "boyfriend": [r"\bboyfriends?\b", r"\bbf\b"],
    "girlfriend": [r"\bgirlfriends?\b", r"\bgf\b"],
    "colleague": [r"\bcolleagues?\b", r"\bcoworkers?\b", r"\bco-workers?\b", r"\bboss\b"],
    "children": [r"\bchild(?:ren)?\b", r"\bkids?\b", r"\bsons?\b", r"\bdaughters?\b", r"\bbaby\b", r"\btoddler\b", r"\bboys?\b", r"\bgirls?\b"],
    "parents": [r"\bparents?\b"],
}

_RECIPIENT_PHRASES = [
    r"\bfor\s+my\s+(?:brother|sister|father|mother|dad|mom|friend|wife|husband|boyfriend|girlfriend|colleague|coworker|kids|children|son|daughter|parents|boss)\b",
    r"\bfor\s+(?:brother|sister|father|mother|dad|mom|friend|wife|husband|boyfriend|girlfriend|colleague|coworker|kids|children|son|daughter|parents|him|her|them)\b",
]

_USE_PHRASES = [
    r"\bfor\s+(?:international\s+travel|business\s+travel|travel(?:ling)?|backpacking|holidays?|vacations?|trekking|hiking|camping|trip|commute|programming|coding|software|running|jogging|gaming|games|office(?:\s+work)?|work|college|school|university|study(?:ing)?|fitness|gym|workout|bad\s+knees)\b",
]

_USE_PATTERNS = {
    "programming": [r"\bprogramm(?:ing)?\b", r"\bcoding\b", r"\bdevelopment\b", r"\bsoftware\b"],
    "gaming": [r"\bgam(?:ing|es?)\b"],
    "international travel": [r"\binternational travel\b", r"\bflights?\b", r"\babroad\b"],
    "business travel": [r"\bbusiness\s+travel\b", r"\bwork\s+trip\b"],
    "holidays": [r"\bholidays?\b", r"\bvacations?\b"],
    "backpacking": [r"\bbackpacking\b"],
    "travel": [r"\btravel(?:ling)?\b", r"\btrip\b", r"\bcommute\b", r"\bcommuting\b"],
    "running": [r"\brunn(?:ing)?\b", r"\bjogg(?:ing)?\b", r"\bmarathon\b"],
    "office": [r"\boffice(?: work)?\b", r"\bdesk job\b", r"\blong working hours\b", r"\bwork from home\b", r"\bwfh\b"],
    "hiking": [r"\bhik(?:ing)?\b", r"\btrail\b"],
    "trekking": [r"\btrek(?:king)?\b"],
    "camping": [r"\bcamp(?:ing)?\b"],
    "walking": [r"\bwalk(?:ing|s)?\b"],
    "college": [r"\bcollege\b", r"\bschool\b", r"\buniversity\b", r"\bstudy(?:ing)?\b"],
    "fitness": [r"\bfitness\b", r"\bgym\b", r"\bworkout\b", r"\bexercise\b"],
}

_OCCASION_PATTERNS = {
    "birthday": [r"\bbirthday\b", r"\bb'day\b"],
    "anniversary": [r"\banniversary\b"],
    "wedding": [r"\bwedding\b", r"\bmarriage\b"],
    "christmas": [r"\bchristmas\b", r"\bx-?mas\b"],
    "diwali": [r"\bdiwali\b", r"\bdeepavali\b"],
    "housewarming": [r"\bhousewarming\b", r"\bnew home\b"],
}

_VAGUE_TERMS = {
    "something", "anything", "stuff", "thing", "things", "item", "items",
    "product", "products", "options", "option", "useful", "nice", "good",
    "best", "top", "great", "brand", "brands", "ones", "one", "different",
    "another", "other", "more", "less", "cheaper", "only", "same", "instead",
    "increase", "decrease", "budget", "relax", "specification", "specifications",
    "specs", "retry", "search", "alternative", "alternatives",
}

_KNOWN_PRODUCTS = [
    ("mobile phone", "mobile phone", "Electronics", "Smartphones"),
    ("smartphone", "smartphone", "Electronics", "Smartphones"),
    ("washing machine", "washing machine", "Home and Kitchen", "Washing Machines"),
    ("skincare products", "skincare products", "Beauty and Personal Care", "Skincare"),
    ("skincare", "skincare", "Beauty and Personal Care", "Skincare"),
    ("travel bag", "travel bag", "Travel and Luggage", "Travel Bags"),
    ("cabin suitcase", "cabin suitcase", "Travel and Luggage", "Suitcases"),
    ("suitcase", "suitcase", "Travel and Luggage", "Suitcases"),
    ("smartwatch", "smartwatch", "Electronics", "Smartwatches"),
    ("backpack", "backpack", "Travel and Luggage", "Backpacks"),
    ("wireless headphones", "wireless headphones", "Electronics", "Headphones"),
    ("headphones", "headphones", "Electronics", "Headphones"),
    ("earbuds", "earbuds", "Electronics", "Headphones"),
    ("earphones", "earphones", "Electronics", "Headphones"),
    ("laptop", "laptop", "Electronics", "Laptops"),
    ("shoes", "shoes", "Fashion and Clothing", "Footwear"),
    ("sneakers", "sneakers", "Fashion and Clothing", "Footwear"),
    ("running shoes", "running shoes", "Fashion and Clothing", "Footwear"),
    ("furniture", "furniture", "Furniture and Home Decor", "Furniture"),
    ("office chair", "office chair", "Furniture and Home Decor", "Office Chairs"),
    ("chair", "chair", "Furniture and Home Decor", "Chairs"),
    ("books", "books", "Books and Stationery", "Books"),
    ("book", "book", "Books and Stationery", "Books"),
    ("programming books", "programming books", "Books and Stationery", "Programming Books"),
    ("microwave", "microwave", "Home and Kitchen", "Kitchen Appliances"),
    ("air fryer", "air fryer", "Home and Kitchen", "Kitchen Appliances"),
    ("coffee maker", "coffee maker", "Home and Kitchen", "Kitchen Appliances"),
    ("refrigerator", "refrigerator", "Home and Kitchen", "Large Appliances"),
    ("t-shirt", "t-shirt", "Fashion and Clothing", "Clothing"),
    ("jacket", "jacket", "Fashion and Clothing", "Clothing"),
    ("gifts", "gifts", "Gifts", "Gifts"),
    ("gift", "gifts", "Gifts", "Gifts"),
    ("present", "gifts", "Gifts", "Gifts"),
    ("presents", "gifts", "Gifts", "Gifts"),
    ("travel equipment", "travel equipment", "Travel and Luggage", "Travel Equipment"),
    ("travel gear", "travel gear", "Travel and Luggage", "Travel Equipment"),
    ("travel accessories", "travel accessories", "Travel and Luggage", "Travel Accessories"),
    ("electronics", "electronics", "Electronics", "Electronics"),
    ("electronic items", "electronics", "Electronics", "Electronics"),
    ("electronic products", "electronics", "Electronics", "Electronics"),
    ("gadgets", "gadgets", "Electronics", "Gadgets"),
    ("sports equipment", "sports equipment", "Sports and Fitness", "Sports Equipment"),
    ("fitness equipment", "fitness equipment", "Sports and Fitness", "Fitness Equipment"),
    ("gym equipment", "gym equipment", "Sports and Fitness", "Gym Equipment"),
    ("workout gear", "workout gear", "Sports and Fitness", "Fitness Equipment"),
    ("clothing", "clothing", "Fashion and Clothing", "Clothing"),
    ("apparel", "apparel", "Fashion and Clothing", "Clothing"),
]

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


def is_broad_shopping_intent(product_type: str | None, category: str | None, text: str = "") -> bool:
    lowered = text.lower()
    pt = str(product_type or "").lower().strip()
    if pt in _BROAD_CATEGORY_TERMS:
        return True
    if _BROAD_GROUP_SUFFIXES.search(pt):
        return True
    if any(term in lowered for term in (
        "travel equipment", "travel gear", "travel accessories", "travel stuff", "travel items",
        "some electronics", "electronic items", "electronic products",
        "sports equipment", "fitness equipment", "workout gear", "gym gear",
        "home appliances", "kitchen appliances", "some furniture", "some clothing"
    )):
        return True
    return False


def extract_flexible_budget(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(
        r"\b(?:any\s+budget|no\s+budget\s*(?:limit)?|no\s+price\s*limit|surprise\s+me|show\s+(?:me\s+)?anything|show\s+anything|whatever\s+budget|budget\s+doesn'?t\s+matter|money\s+is\s+no\s+object|open\s+budget|any\s+price|without\s+budget|flexible\s+budget)\b",
        lowered,
        re.I
    ))


def extract_size(text: str) -> str | None:
    lowered = text.lower()
    match = re.search(
        r"\b(\d+\s*(?:l|litres?|liters?|ltrs?)|cabin(?:\s+size)?|check-?in|carry-?on|55\s*cm|65\s*cm|75\s*cm|\d+\s*inch|medium|large|small|compact|xl|xxl)\b",
        lowered,
        re.I
    )
    if match:
        return match.group(0).strip().lower()
    return None

_FOLLOWUP_INDICATORS = re.compile(
    r"\b(under|below|less|more|cheaper|cheap|max|min|different|another|other|brands?|only|also|same|instead|ones?|it|them|similar|make|size|color|colour|with|without|show|prefer)\b",
    re.I
)

_INTENT_SHIFT_PATTERNS = [
    r"\b(?:i\s+changed\s+my\s+mind|change\s+my\s+mind)\b",
    r"\binstead\s+suggest\s+a\b",
    r"\bactually\s+(?:show\s+me|suggest|find|look\s+for)\b",
    r"\bforget\s+(?:that|about\s+it)\b",
    r"\bstart\s+over\b",
    r"\bnew\s+search\b",
]


def extract_recipient(text: str) -> str | None:
    lowered = text.lower()
    for recipient, patterns in _RECIPIENT_PATTERNS.items():
        if any(re.search(pat, lowered) for pat in patterns):
            return recipient
    return None


def extract_intended_use(text: str) -> str | None:
    lowered = text.lower()
    for use, patterns in _USE_PATTERNS.items():
        if any(re.search(pat, lowered) for pat in patterns):
            return use
    return None


def extract_occasion(text: str) -> str | None:
    lowered = text.lower()
    for occasion, patterns in _OCCASION_PATTERNS.items():
        if any(re.search(pat, lowered) for pat in patterns):
            return occasion
    return None


def extract_budget(text: str) -> tuple[float | None, float | None]:
    lowered = text.lower()
    range_match = re.search(rf"\b(?:between\s*)?{_CUR}{_AMT}\s*(?:to|and|-)\s*{_CUR}{_AMT}\b", lowered)
    if range_match:
        has_curr = bool(re.search(r"₹|rs|inr|rupee", range_match.group(0)))
        has_between = "between" in lowered
        if has_curr or has_between or range_match.group(2) or range_match.group(4):
            def _amt(n, s):
                v = float(n.replace(",", ""))
                return v * 1000 if s and s.lower() == "k" else v
            return _amt(range_match.group(1), range_match.group(2)), _amt(range_match.group(3), range_match.group(4))
    match = _MAX_RE.search(lowered)
    if match:
        v = float(match.group(1).replace(",", ""))
        return None, v * 1000 if match.group(2) and match.group(2).lower() == "k" else v
    match = _MIN_RE.search(lowered)
    if match:
        v = float(match.group(1).replace(",", ""))
        return v * 1000 if match.group(2) and match.group(2).lower() == "k" else v, None
    bare = _BARE_BUDGET_RE.match(lowered.strip())
    if bare:
        v = float(bare.group(1).replace(",", ""))
        return None, v * 1000 if bare.group(2) and bare.group(2).lower() == "k" else v
    return None, None


def clean_for_product(text: str) -> str:
    cleaned = text.lower()
    cleaned = _RANGE_RE.sub(" ", cleaned)
    cleaned = _MIN_RE.sub(" ", cleaned)
    cleaned = _MAX_RE.sub(" ", cleaned)
    cleaned = _BARE_BUDGET_RE.sub(" ", cleaned)
    for pat in _RECIPIENT_PHRASES:
        cleaned = re.sub(pat, " ", cleaned, flags=re.I)
    for pat in _USE_PHRASES:
        cleaned = re.sub(pat, " ", cleaned, flags=re.I)
    for pat in _COMMAND_PHRASES:
        cleaned = re.sub(pat, " ", cleaned, flags=re.I)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def product_phrase(lowered: str) -> str:
    for pattern, canonical, _, _ in _KNOWN_PRODUCTS:
        if re.search(rf"\b{re.escape(pattern)}\b", lowered):
            return canonical

    cleaned = clean_for_product(lowered)
    words = cleaned.split()
    tech_terms = {
        "wireless", "wired", "anc", "bluetooth", "lightweight", "waterproof",
        "durable", "comfortable", "noise", "cancellation", "cancelling",
        "fast", "charging", "ram", "ssd", "intel", "ryzen"
    }
    content_words = [
        w for w in words
        if w not in _VAGUE_TERMS
        and w not in tech_terms
        and w not in {"for", "in", "with", "without", "and", "or", "to", "my", "a", "an", "the", "under", "below", "above"}
    ]
    if not content_words:
        return ""
    return " ".join(content_words[:3])


def _canonical_category(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    aliases = {
        "laptops": "Electronics",
        "laptop": "Electronics",
        "computer": "Electronics",
        "computers": "Electronics",
        "electronics": "Electronics",
        "laptops and electronics": "Electronics",
        "backpack": "Travel and Luggage",
        "backpacks": "Travel and Luggage",
        "travel bag": "Travel and Luggage",
        "travel and luggage": "Travel and Luggage",
        "travel equipment": "Travel and Luggage",
        "travel gear": "Travel and Luggage",
        "travel accessories": "Travel and Luggage",
        "travel stuff": "Travel and Luggage",
        "travel items": "Travel and Luggage",
        "luggage": "Travel and Luggage",
        "suitcases": "Travel and Luggage",
        "suitcase": "Travel and Luggage",
        "electronics": "Electronics",
        "electronic": "Electronics",
        "electronic items": "Electronics",
        "electronic products": "Electronics",
        "gadget": "Electronics",
        "gadgets": "Electronics",
        "appliances": "Home and Kitchen",
        "home appliances": "Home and Kitchen",
        "kitchen appliances": "Home and Kitchen",
        "skin care": "Beauty and Personal Care",
        "skincare": "Beauty and Personal Care",
        "skincare products": "Beauty and Personal Care",
        "beauty": "Beauty and Personal Care",
        "beauty products": "Beauty and Personal Care",
        "beauty and personal care": "Beauty and Personal Care",
        "fitness": "Sports and Fitness",
        "gym": "Sports and Fitness",
        "sports": "Sports and Fitness",
        "sports and fitness": "Sports and Fitness",
        "sports equipment": "Sports and Fitness",
        "sports gear": "Sports and Fitness",
        "fitness equipment": "Sports and Fitness",
        "fitness gear": "Sports and Fitness",
        "gym equipment": "Sports and Fitness",
        "gym gear": "Sports and Fitness",
        "workout gear": "Sports and Fitness",
        "gift": "Gifts",
        "gifts": "Gifts",
        "shoes": "Fashion and Clothing",
        "footwear": "Fashion and Clothing",
        "clothing": "Fashion and Clothing",
        "apparel": "Fashion and Clothing",
        "fashion and clothing": "Fashion and Clothing",
        "furniture": "Furniture and Home Decor",
        "furniture and home decor": "Furniture and Home Decor",
        "home and kitchen": "Home and Kitchen",
        "kitchen": "Home and Kitchen",
        "books": "Books and Stationery",
        "books and stationery": "Books and Stationery",
    }
    if text in aliases:
        return aliases[text]
    # Default capitalization with lower 'and'
    words = [w.capitalize() if w != "and" else "and" for w in text.split()]
    return " ".join(words)


def _canonical_subcategory(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    aliases = {
        "laptop": "Laptops",
        "laptops": "Laptops",
        "notebook computer": "Laptops",
        "wireless headphones": "Headphones",
        "headphones": "Headphones",
        "earbuds": "Headphones",
        "running shoes": "Running",
        "running": "Running",
        "shoes": "Footwear",
        "office chair": "Office Chairs",
        "microwave": "Kitchen Appliances",
        "cabin suitcase": "Suitcases",
        "suitcase": "Suitcases",
        "backpack": "Backpacks",
        "programming books": "Programming Books",
        "smartwatch": "Smartwatches",
        "mobile phone": "Smartphones",
        "smartphone": "Smartphones",
        "washing machine": "Washing Machines",
        "skincare": "Skincare",
        "gifts": "Gifts",
    }
    return aliases.get(text, str(value).strip())


def _detect_brand_preferences(text: str) -> tuple[list[str], list[str], bool]:
    lowered = text.lower()
    brand_diversity = bool(re.search(r"\b(?:different|other|another|variety\s+of|mix\s+of|more)\s+brands?\b", lowered))
    
    known_brands = [
        "sony", "apple", "dell", "hp", "lenovo", "asus", "acer", "samsung",
        "boat", "noise", "boult", "jbl", "bose", "sennheiser", "oneplus",
        "xiaomi", "realme", "nike", "adidas", "puma", "safari",
        "american tourister", "vip", "skybags", "lg", "whirlpool", "ifb",
        "bosch", "philips", "logitech",
    ]
    preferred = []
    excluded = []
    
    # Check exclusions e.g. "not boat", "exclude samsung", "no hp"
    exclude_match = re.findall(r"\b(?:not|exclude|no|except|without)\s+([a-z\s]+?)(?:\band\b|,|$)", lowered)
    for clause in exclude_match:
        for brand in known_brands:
            if re.search(rf"\b{re.escape(brand)}\b", clause):
                excluded.append(brand)
                
    for brand in known_brands:
        if brand not in excluded and re.search(rf"\b{re.escape(brand)}\b", lowered):
            preferred.append(brand)
            
    return list(dict.fromkeys(preferred)), list(dict.fromkeys(excluded)), brand_diversity


def _detect_technical_preferences(text: str) -> list[str]:
    lowered = text.lower()
    specs = []
    spec_terms = {
        "anc": ("anc", "noise cancellation", "active noise cancellation", "noise cancelling"),
        "wireless": ("wireless", "bluetooth"),
        "wired": ("wired",),
        "lightweight": ("lightweight", "light weight", "portable"),
        "comfortable": ("comfortable", "comfy", "comfort", "cushioning", "joint support", "bad knees"),
        "waterproof": ("waterproof", "water resistant"),
        "durable": ("durable", "sturdy", "long lasting"),
        "16gb ram": ("16gb ram", "16 gb ram", "16gb"),
        "512gb ssd": ("512gb ssd", "512 gb ssd", "512gb"),
        "i5": ("core i5", "intel i5", "i5"),
        "i7": ("core i7", "intel i7", "i7"),
        "fast charging": ("fast charging", "quick charge"),
        "front load": ("front load", "front-load"),
        "top load": ("top load", "top-load"),
    }
    for spec_name, terms in spec_terms.items():
        if any(term in lowered for term in terms):
            specs.append(spec_name)
    return specs


def _deterministic_intent(text: str) -> dict:
    lowered = text.lower()
    budget_min, budget_max = extract_budget(text)
    flexible_budget = extract_flexible_budget(text)
    if flexible_budget:
        budget_min = None
        budget_max = None
    size = extract_size(text)
    recipient = extract_recipient(text)
    intended_use = extract_intended_use(text)
    occasion = extract_occasion(text)
    prod_phrase = product_phrase(lowered)
    
    # Determine category & subcategory from lexicon or product phrase
    category = None
    subcategory = None
    product_type = prod_phrase or None

    for pattern, canonical, cat, subcat in _KNOWN_PRODUCTS:
        if re.search(rf"\b{re.escape(pattern)}\b", lowered):
            product_type = canonical
            category = cat
            subcategory = subcat
            break

    if not category and product_type:
        if product_type in ("gift", "gifts"):
            category = "Gifts"
            subcategory = "Gifts"
        else:
            category = _canonical_category(product_type)

    pref_brands, excl_brands, brand_diversity = _detect_brand_preferences(text)
    tech_specs = _detect_technical_preferences(text)
    if size and size not in tech_specs:
        tech_specs.append(size)
    
    is_broad = is_broad_shopping_intent(product_type, category, text)

    hard_constraints = []
    if "in stock" in lowered or "available" in lowered:
        hard_constraints.append("in_stock")

    return {
        "original_query": text,
        "category": _canonical_category(category),
        "subcategory": _canonical_subcategory(subcategory),
        "product_type": product_type,
        "recipient": recipient,
        "intended_use": intended_use,
        "activity": intended_use,  # Backwards compatibility
        "occasion": occasion,
        "budget_min": budget_min,
        "budget_max": budget_max,
        "flexible_budget": flexible_budget,
        "size": size,
        "is_broad_category": is_broad,
        "preferred_brands": pref_brands,
        "excluded_brands": excl_brands,
        "brand_diversity": brand_diversity,
        "preferences": tech_specs,
        "technical_specifications": tech_specs,
        "hard_constraints": hard_constraints,
        "is_followup": False,
    }


def extract_intent(state: dict) -> dict:
    text = state.get("normalized_query", state.get("message", "")).strip()
    previous_constraints = dict(state.get("constraints", {}))
    lowered = text.lower()
    
    # Check for empty query
    if not text:
        return {
            "constraints": previous_constraints,
            "intent_confidence": 0.0,
            "intent_source": "deterministic",
            "query_is_new": False,
        }

    deterministic = _deterministic_intent(text)

    # Detect fallback suggestion acceptance
    is_fallback_action = bool(
        re.search(r"\b(?:increase\s+budget|include\s+other\s+brands|show\s+all\s+brands|clear\s+brand\s+filter|relax\s+specifications|relax\s+specs|retry)\b", lowered)
    )

    # 1. Detect Intent Shift (User started a genuinely new search)
    has_explicit_shift = any(re.search(pat, lowered) for pat in _INTENT_SHIFT_PATTERNS)
    # If a brand new product type is mentioned with NO follow-up markers and different from previous
    prev_prod = previous_constraints.get("product_type")
    new_prod = deterministic.get("product_type")
    is_distinct_product = bool(
        new_prod
        and prev_prod
        and new_prod != prev_prod
        and not is_fallback_action
        and not any(phrase in lowered for phrase in ("same", "similar", "also", "instead", "under", "cheaper", "brand"))
    )
    is_new_shopping_intent = (has_explicit_shift or is_distinct_product) and not is_fallback_action

    # 2. Try LLM Extraction
    prompt = f"""You are an expert intent extractor for a conversational shopping assistant.
Current customer message: {text!r}
Previous conversation constraints: {previous_constraints!r}

Extract shopping intent accurately into JSON.
Return JSON ONLY with these exact keys:
- category: string or null
- subcategory: string or null
- product_type: string or null
- recipient: string or null (e.g. brother, sister, father, mother, friend, wife, husband, colleague, children)
- intended_use: string or null (e.g. programming, running, travel, gaming, office work)
- occasion: string or null (e.g. birthday, wedding, anniversary)
- budget_min: number or null
- budget_max: number or null
- preferred_brands: list of strings
- excluded_brands: list of strings
- brand_diversity: boolean (true if user wants to see different or other brands)
- size: string or null
- color: string or null
- quantity: integer or null
- preferences: list of strings
- technical_specifications: list of strings
- hard_constraints: list of strings (e.g. ["in_stock"])
- is_followup: boolean (true if this message modifies or follows up on the previous request)
- is_new_shopping_intent: boolean (true if customer changed topic or started a new product request)

STRICT RULES:
1. NEVER invent a budget, brand, recipient, or specification not explicitly stated by the user.
2. If the user asks for a gift for their brother, recipient is "brother". NEVER change to sister or anyone else.
3. If unspecified, use null or empty list."""

    llm_result = openrouter_json(prompt, operation="extract_intent")
    extracted = dict(llm_result) if isinstance(llm_result, dict) else {}

    # Merge LLM result with deterministic guarantees
    if is_new_shopping_intent or extracted.get("is_new_shopping_intent"):
        constraints: dict[str, Any] = {"original_query": text}
        is_followup = False
    else:
        constraints = previous_constraints.copy()
        # Detect if this is a follow-up
        is_followup = bool(
            is_fallback_action
            or (
                previous_constraints
                and (
                    extracted.get("is_followup")
                    or bool(_FOLLOWUP_INDICATORS.search(lowered))
                    or not deterministic.get("product_type")
                    or len(text.split()) <= 4
                )
            )
        )

    constraints["is_followup"] = is_followup

    # Recipient: preserve existing unless explicitly updated
    detected_recipient = extracted.get("recipient") or deterministic.get("recipient")
    if detected_recipient:
        constraints["recipient"] = detected_recipient.strip().lower()

    # Product Type, Category, Subcategory
    detected_product = extracted.get("product_type") or deterministic.get("product_type")
    if detected_product and (not is_followup or not previous_constraints.get("product_type")):
        constraints["product_type"] = detected_product
    elif not constraints.get("product_type") and detected_product:
        constraints["product_type"] = detected_product

    detected_cat = extracted.get("category") or deterministic.get("category")
    if detected_cat and (not is_followup or not previous_constraints.get("category")):
        constraints["category"] = _canonical_category(detected_cat)

    detected_subcat = extracted.get("subcategory") or deterministic.get("subcategory")
    if detected_subcat and (not is_followup or not previous_constraints.get("subcategory")):
        constraints["subcategory"] = _canonical_subcategory(detected_subcat)

    # Intended Use / Activity
    detected_use = extracted.get("intended_use") or deterministic.get("intended_use")
    if detected_use:
        constraints["intended_use"] = detected_use
        constraints["activity"] = detected_use

    # Occasion
    detected_occasion = extracted.get("occasion") or deterministic.get("occasion")
    if detected_occasion:
        constraints["occasion"] = detected_occasion

    # Budget handling
    det_min, det_max = deterministic.get("budget_min"), deterministic.get("budget_max")
    llm_min = extracted.get("budget_min")
    llm_max = extracted.get("budget_max")
    
    new_budget_min = det_min if det_min is not None else (float(llm_min) if llm_min is not None else None)
    new_budget_max = det_max if det_max is not None else (float(llm_max) if llm_max is not None else None)

    # Handle fallback suggestion: "Increase budget to ₹25,000" or "Increase budget by 20%"
    inc_match = re.search(rf"\bincrease\s+budget(?:\s+to)?\s*{_CUR}{_AMT}\b", lowered)
    if inc_match:
        v = float(inc_match.group(1).replace(",", ""))
        new_budget_max = v * 1000 if inc_match.group(2) and inc_match.group(2).lower() == "k" else v
    elif "increase budget by 20%" in lowered:
        if constraints.get("budget_max"):
            new_budget_max = round(constraints["budget_max"] * 1.2, -2)

    # Handle "make it cheaper" / "show something cheaper"
    is_cheaper = bool(re.search(r"\b(?:cheaper|make it cheaper|show something cheaper|less expensive|lower price)\b", lowered))
    if is_cheaper:
        current_max = constraints.get("budget_max")
        if current_max is not None:
            # Reduce budget by ~20%
            constraints["budget_max"] = round(current_max * 0.8)
        constraints["cheaper"] = True
    elif new_budget_max is not None:
        constraints["budget_max"] = new_budget_max
        
    if new_budget_min is not None:
        constraints["budget_min"] = new_budget_min

    # Handle fallback suggestion: "Include other brands" / "Show all brands"
    if re.search(r"\b(?:include\s+other\s+brands|show\s+all\s+brands|clear\s+brand\s+filter)\b", lowered):
        constraints["excluded_brands"] = []
        constraints["preferred_brands"] = []
        constraints["brand_diversity"] = True

    # Handle fallback suggestion: "Relax specifications"
    if re.search(r"\b(?:relax\s+specifications|clear\s+specifications|any\s+specs)\b", lowered):
        constraints["preferences"] = []
        constraints["technical_specifications"] = []

    # Brand Diversity & Brand Preferences
    if extracted.get("brand_diversity") or deterministic.get("brand_diversity"):
        constraints["brand_diversity"] = True
        
    pref_brands = list(dict.fromkeys(
        constraints.get("preferred_brands", [])
        + _as_list(extracted.get("preferred_brands"))
        + deterministic.get("preferred_brands", [])
    ))
    if pref_brands:
        constraints["preferred_brands"] = pref_brands

    excl_brands = list(dict.fromkeys(
        constraints.get("excluded_brands", [])
        + _as_list(extracted.get("excluded_brands"))
        + deterministic.get("excluded_brands", [])
    ))
    if excl_brands:
        constraints["excluded_brands"] = excl_brands

    # Specifications and Preferences
    all_prefs = list(dict.fromkeys(
        constraints.get("preferences", [])
        + _as_list(extracted.get("preferences"))
        + _as_list(extracted.get("technical_specifications"))
        + deterministic.get("preferences", [])
    ))
    if all_prefs:
        constraints["preferences"] = all_prefs
        constraints["technical_specifications"] = all_prefs

    # Hard constraints
    hard_c = list(dict.fromkeys(
        constraints.get("hard_constraints", [])
        + _as_list(extracted.get("hard_constraints"))
        + deterministic.get("hard_constraints", [])
    ))
    if hard_c:
        constraints["hard_constraints"] = hard_c

    # Flexible budget
    if deterministic.get("flexible_budget") or extracted.get("flexible_budget"):
        constraints["flexible_budget"] = True
        constraints["budget_max"] = None
        constraints["budget_min"] = None

    # Size & Color
    detected_size = extracted.get("size") or deterministic.get("size")
    if detected_size:
        constraints["size"] = str(detected_size).strip().lower()
        if constraints["size"] not in constraints.get("preferences", []):
            constraints.setdefault("preferences", []).append(constraints["size"])
    if extracted.get("color"):
        constraints["color"] = str(extracted["color"]).strip()

    # Broad category flag
    if is_new_shopping_intent or "is_broad_category" not in constraints:
        constraints["is_broad_category"] = deterministic.get("is_broad_category", False)
    # If a specific concrete product type is detected, clear is_broad_category
    if detected_product and not is_broad_shopping_intent(detected_product, constraints.get("category"), text):
        constraints["is_broad_category"] = False
    elif deterministic.get("is_broad_category"):
        constraints["is_broad_category"] = True

    if not constraints.get("original_query"):
        constraints["original_query"] = text

    # Tool Intent Detection (Comparison, Reviews, Alternatives)
    tool_intent = None
    alternative_type = None
    tool_target = None

    if re.search(r"\b(?:compare|comparison|difference(?:s)?\s+between|versus|vs\.?)\b", lowered):
        tool_intent = "compare"
        if re.search(r"\b(?:two|both|top\s*2|2)\b", lowered):
            tool_target = "top_2"
        elif re.search(r"\b(?:three|top\s*3|3)\b", lowered):
            tool_target = "top_3"
        else:
            tool_target = "top_3"

    elif re.search(r"\b(?:reviews?|customer\s+feedback|user\s+opinions?|what\s+do\s+(?:users|people|customers)\s+say|pros\s+and\s+cons|reliability)\b", lowered):
        tool_intent = "reviews"
        if re.search(r"\b(?:first|1st|top)\b", lowered):
            tool_target = "first"
        elif re.search(r"\b(?:second|2nd)\b", lowered):
            tool_target = "second"
        elif re.search(r"\b(?:third|3rd)\b", lowered):
            tool_target = "third"

    elif re.search(r"\b(?:alternatives?|other\s+options?)\b", lowered) or (
        state.get("previous_products")
        and re.search(r"\b(?:cheaper\s+options?|different\s+brands?|other\s+brands?|similar\s+options?|budget\s+friendly)\b", lowered)
    ):
        tool_intent = "alternatives"
        if re.search(r"\b(?:cheaper|lower\s+price|less\s+expensive)\b", lowered):
            alternative_type = "cheaper"
        elif re.search(r"\b(?:different\s+brands?|other\s+brands?)\b", lowered):
            alternative_type = "different_brands"
        elif re.search(r"\b(?:budget|budget\s+friendly)\b", lowered):
            alternative_type = "budget_friendly"
        elif re.search(r"\b(?:better\s+specs?|better\s+specifications?|upgrade|higher\s+end)\b", lowered):
            alternative_type = "better_specifications"
        else:
            alternative_type = "similar"

    elif re.search(r"\b(?:why.*ranked|why.*first|why.*#1|why.*better|why.*recommended|what.*(?:not|unmet|missing|lacks?)|doesn't\s*satisfy)\b", lowered):
        tool_intent = "explain_ranking"

    return {
        "constraints": constraints,
        "intent_confidence": 0.95 if llm_result else 0.75,
        "intent_source": "openrouter" if llm_result else "deterministic",
        "query_is_new": is_new_shopping_intent,
        "tool_intent": tool_intent,
        "tool_target": tool_target,
        "alternative_type": alternative_type,
    }
