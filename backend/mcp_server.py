"""MCP v2 tools backed by the same configured product providers as FastAPI."""

import asyncio

from mcp.server.mcpserver import MCPServer

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


if __name__ == "__main__":
    asyncio.run(mcp.run_stdio_async())