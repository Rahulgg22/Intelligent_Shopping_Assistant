from __future__ import annotations

from typing import Protocol

from backend.schemas.product import Product, ProductSearchResult


class ProviderError(RuntimeError):
    """A provider could not complete a product operation."""


class ProviderStatus(Protocol):
    name: str

    def status(self) -> dict[str, str]: ...


class ProductProvider(Protocol):
    name: str

    def status(self) -> dict[str, str]: ...

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
    ) -> ProductSearchResult: ...

    def get(self, product_id: str) -> Product | None: ...