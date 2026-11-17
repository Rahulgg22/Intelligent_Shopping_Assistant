"""MCP v2 tools backed by the same configured product providers as FastAPI."""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.mcpserver import MCPServer

from backend.agents.alternatives import find_product_alternatives_impl
from backend.agents.comparison import compare_products_impl
from backend.agents.product_reviews import search_product_reviews_impl
from backend.services.product_sources import build_product_service

build_catalog_service = build_product_service


mcp = MCPServer("ps12-shopping", version="1.0.0")


@mcp.tool()
def search_products(
    query: str = "",
    category: str | None = None,
    subcategory: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    in_stock_only: bool = True,
    limit: int = 20,
) -> dict:
    """Search configured product sources with deterministic budget and stock filters."""
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    result = build_product_service().search(
        query,
        category=category,
        subcategory=subcategory,
        min_price=min_price,
        max_price=max_price,
        in_stock_only=in_stock_only,
        limit=limit,
    )
    payload = result.model_dump(mode="json")
    payload["retrieval_errors"] = payload["errors"]
    return payload


@mcp.tool()
def get_product_details(product_id: str) -> dict:
    """Return one normalized product from configured sources, or an explicit not-found result."""
    product = build_product_service().get(product_id)
    return {"product": product.model_dump(mode="json") if product else None}


@mcp.tool()
def get_provider_status() -> dict:
    """Return readiness for every configured product source."""
    service = build_product_service()
    return {"providers": {provider.name: provider.status() for provider in service.provider.providers}}


@mcp.tool()
def compare_products(
    products: list[dict[str, Any]],
    user_requirements: list[str] | None = None,
) -> dict:
    """Compare two or more products against prices, specifications, and user requirements."""
    if not isinstance(products, list):
        raise ValueError("products must be a list of product dictionaries")
    return compare_products_impl(products=products, user_requirements=user_requirements)


@mcp.tool()
def search_product_reviews(
    product_name: str,
    brand: str = "",
    search_context: str = "",
    max_results: int = 8,
) -> dict:
    """Find independent reviews, editorial verdicts, and customer feedback for a product."""
    if not product_name or not str(product_name).strip():
        raise ValueError("product_name must not be empty")
    return search_product_reviews_impl(
        product_name=product_name,
        brand=brand,
        search_context=search_context,
        max_results=max_results,
    )


@mcp.tool()
def find_product_alternatives(
    original_product: dict[str, Any],
    alternative_type: str = "similar",
    user_requirements: dict[str, Any] | None = None,
    max_results: int = 8,
) -> dict:
    """Find real external alternative products based on type (cheaper, different_brands, similar, better_specifications, budget_friendly)."""
    if not isinstance(original_product, dict):
        raise ValueError("original_product must be a dictionary")
    return find_product_alternatives_impl(
        original_product=original_product,
        alternative_type=alternative_type,
        user_requirements=user_requirements,
        max_results=max_results,
    )


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())
