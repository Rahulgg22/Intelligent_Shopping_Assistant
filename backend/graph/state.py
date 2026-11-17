from typing import TypedDict, Any


class ShoppingState(TypedDict, total=False):

    session_id: str

    message: str

    normalized_query: str

    effective_query: str

    turn: int

    constraints: dict[str, Any]

    intent_confidence: float

    needs_clarification: bool

    clarification_decision: str

    clarifying_question: str | None

    pending_clarification: str | None

    pending_clarification_question: str | None

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

    search_status: str

    fallback_reason: str | None

    fallback_suggestions: list[str]

    rejection_reasons: dict[str, Any]

    # Tool and multi-turn MCP execution state
    previous_products: list[dict]

    tool_intent: str | None

    tool_target: str | None

    alternative_type: str | None

    selected_tool: str | None

    comparison_result: dict[str, Any] | None

    review_result: dict[str, Any] | None

    alternatives_result: dict[str, Any] | None
