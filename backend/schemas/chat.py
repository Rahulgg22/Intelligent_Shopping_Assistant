from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class ProductOut(BaseModel):
    product_id: str
    name: str
    category: str
    subcategory: str = ""
    brand: str = ""
    price: float | None = None
    currency: str | None = "INR"
    description: str
    features: list[str] = []
    use_cases: list[str] = []
    tags: list[str] = []
    rating: float | None = None
    in_stock: bool | None = True
    fit_score: float | None = None
    explanation: str | None = None
    tradeoffs: list[str] = []
    original_price: float | None = None
    product_url: str | None = None
    source_name: str = "local"
    source_product_id: str | None = None
    availability: str = "unknown"
    price_verified: bool = False
    availability_verified: bool = False
    retrieved_at: str | None = None
    match_reasons: list[str] = []
    review_count: int | None = None
    stock_quantity: int | None = None
    discount_percent: float | None = None
    image_url: str | None = None
    weight_kg: float | None = None
    color: str = ""
    warranty_months: int | None = None
    specifications: str = ""
    rank: int | None = None
    matched_requirements: list[str] = []
    benefits: list[str] = []
    limitations: list[str] = []
    comparison_note: str | None = None
    retrieval_reason: str | None = None
    ranking_reason: str | None = None
    requirement_matches: list[dict[str, Any]] = []
    strengths: list[str] = []
    matched_attributes: list[str] = []
    unverified_attributes: list[str] = []
    explanation_details: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    session_id: str
    action: Literal["recommendations", "clarification", "alternatives", "message"]
    message: str
    products: list[ProductOut] = []
    clarifying_question: str | None = None
    active_constraints: dict[str, Any] = {}
    provider_status: dict[str, str] = {}
    retrieval_errors: dict[str, str] = {}
    search_status: str = "success"
    fallback_suggestions: list[str] = []
    metadata: dict[str, Any] = {}