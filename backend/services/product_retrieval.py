from __future__ import annotations

from backend.providers.base import ProductProvider
from backend.schemas.product import Product, ProductSearchResult


class ProductRetrievalService:
    def __init__(self, provider: ProductProvider):
        self.provider = provider

    def search(
        self,
        query: str = "",
        *,
        category: str | None = None,
        subcategory: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        in_stock_only: bool = False,
        limit: int = 50,
    ) -> ProductSearchResult:
        return self.provider.search(
            query,
            category=category,
            subcategory=subcategory,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=in_stock_only,
            limit=limit,
        )

    def get(self, product_id: str) -> Product | None:
        return self.provider.get(product_id)

    def status(self) -> dict[str, str]:
        return self.provider.status()