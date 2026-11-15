from typing import TypedDict, Any


class ShoppingState(TypedDict, total=False):

    session_id: str

    message: str

    normalized_query: str

    turn: int

    constraints: dict[str, Any]

    intent_confidence: float

    needs_clarification: bool

    clarifying_question: str | None

    mapped_filters: dict[str, Any]

    candidates: list[dict]

    ranked_products: list[dict]

    recommended_products: list[dict]

    fallback_products: list[dict]

    history: list[dict]

    action: str

    response_message: str

    alternatives: list[dict]

    query_is_new: bool

    intent_source: str

    provider_status: dict[str, str]

    retrieval_errors: dict[str, str]