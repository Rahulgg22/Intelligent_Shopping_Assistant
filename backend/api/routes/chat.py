from fastapi import APIRouter

from backend.agents.orchestrator import process_message
from backend.schemas.chat import ChatRequest, ChatResponse


router = APIRouter(prefix="/api/v1", tags=["Shopping Chat"])


def parse_list(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def normalize_product(product):
    normalized = dict(product)
    for field in ("features", "use_cases", "tags", "tradeoffs", "matched_requirements", "benefits", "limitations", "strengths", "matched_attributes", "unverified_attributes"):
        normalized[field] = parse_list(normalized.get(field, []))
    price = normalized.get("price")
    normalized["price"] = float(price) if price not in (None, "") else None
    rating = normalized.get("rating")
    normalized["rating"] = float(rating) if rating not in (None, "") else None
    availability = normalized.get("availability", "unknown")
    normalized["in_stock"] = (
        True if availability == "in_stock" and normalized.get("availability_verified") is True
        else False if availability == "out_of_stock"
        else None
    )
    if "rank" in normalized and normalized["rank"] is not None:
        normalized["rank"] = int(normalized["rank"])
    return normalized


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = process_message(request.session_id, request.message)
    products = result.get("recommended_products") or result.get("fallback_products") or []
    return ChatResponse(
        session_id=request.session_id,
        action=result.get("action", "message"),
        message=result.get("response_message", ""),
        products=[normalize_product(product) for product in products],
        clarifying_question=result.get("clarifying_question"),
        active_constraints=result.get("constraints", {}),
        provider_status=result.get("provider_status", {}),
        retrieval_errors=result.get("retrieval_errors", {}),
        search_status=result.get("search_status", "success"),
        fallback_suggestions=result.get("fallback_suggestions", []),
        metadata={
            "turn": result.get("turn", 0),
            "history": result.get("history", []),
            "intent_source": result.get("intent_source"),
            "fallback_reason": result.get("fallback_reason"),
            "selected_tool": result.get("selected_tool"),
            "comparison": result.get("comparison_result"),
            "reviews": result.get("review_result"),
            "alternatives": result.get("alternatives_result"),
            "notes": [value for value in result.get("retrieval_errors", {}).values() if "verified" in value.lower()],
        },
    )