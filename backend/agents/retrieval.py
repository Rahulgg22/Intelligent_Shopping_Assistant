"""Compatibility entry point for the external provider retrieval service."""

from backend.services.catalog import build_catalog_service


def retrieve_candidates(state):
    filters = state.get("mapped_filters", {})
    result = build_catalog_service().search(
        state.get("normalized_query", state.get("message", "")),
        category=filters.get("category"),
        subcategory=filters.get("subcategory"),
        max_price=filters.get("max_price"),
        in_stock_only=filters.get("in_stock_only", True),
        limit=filters.get("limit", 50),
    )
    return {
        "candidates": [product.model_dump(mode="json") for product in result.products],
        "provider_status": result.provider_status,
        "retrieval_errors": result.errors,
    }
