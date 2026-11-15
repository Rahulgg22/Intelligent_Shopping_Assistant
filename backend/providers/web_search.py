from __future__ import annotations

import re
from typing import Any

import httpx

from backend.config import settings
from backend.providers.base import ProviderError
from backend.providers.page_extractor import PageExtractor
from backend.schemas.product import Product, ProductSearchResult


class TavilyWebSearchProvider:
    """Tavily discovery adapter with optional permitted-page enrichment."""

    name = "tavily_web_search"
    discovery = True

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        self.api_key = settings.tavily_api_key if api_key is None else api_key
        self.endpoint = settings.tavily_endpoint if endpoint is None else endpoint
        self.extractor = PageExtractor(timeout=settings.tavily_timeout_seconds)

    def status(self) -> dict[str, str]:
        if not self.api_key:
            return {"name": self.name, "state": "missing_credentials"}
        return {"name": self.name, "state": "ready"}

    def search(self, query: str = "", *, category: str | None = None,
               subcategory: str | None = None, min_price: float | None = None,
               max_price: float | None = None, in_stock_only: bool = False,
               limit: int = 50, **_: Any) -> ProductSearchResult:
        if not self.api_key:
            raise ProviderError("Tavily API key is not configured")
        focused_query = " ".join(part for part in (category, subcategory, query) if part).strip()
        payload = {
            "query": focused_query,
            "search_depth": settings.tavily_search_depth,
            "max_results": min(max(limit, 1), settings.tavily_max_results),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        try:
            response = httpx.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=settings.tavily_timeout_seconds,
            )
            if response.status_code == 401:
                raise ProviderError("Tavily credentials were rejected")
            if response.status_code == 429:
                raise ProviderError("Tavily rate limit reached")
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderError("Tavily request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Tavily search failed: {exc}") from exc

        products = []
        seen_urls = set()
        for item in data.get("results", []):
            url = item.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            searchable = " ".join(str(item.get(field, "")) for field in ("title", "content", "snippet")).lower()
            category_terms = [term.lower() for term in (category or "").split() if term.isalpha() and len(term) > 3 and term.lower() not in {"missing", "unknown", "category"}]
            subcategory_terms = [term.lower() for term in (subcategory or "").split() if term.isalpha() and len(term) > 3]
            if category and not category_terms and not subcategory_terms:
                continue
            if (category_terms or subcategory_terms) and not any(
                re.search(rf"\b{re.escape(term)}\b", searchable)
                for term in category_terms + subcategory_terms
            ):
                continue
            try:
                discovery = self._discovery_product(item, category, subcategory)
            except ValueError:
                continue
            enriched = self.extractor.extract(url, fallback=discovery) if len(products) < settings.tavily_page_fetch_limit else None
            candidate = enriched or discovery
            if in_stock_only and candidate.availability == "out_of_stock":
                continue
            if (max_price is not None or min_price is not None) and (
                not candidate.price_verified
                or candidate.price is None
                or (candidate.currency or "").upper() != "INR"
            ):
                continue
            if max_price is not None and candidate.price > max_price:
                continue
            if min_price is not None and candidate.price < min_price:
                continue
            products.append(candidate)

        result = ProductSearchResult(
            products=products[:limit],
            providers=[self.name],
            provider_status={self.name: "success_with_results" if products else "success_with_no_results"},
        )
        if max_price is not None and not products:
            result.errors[self.name] = "Relevant pages were found, but no current prices were verified for the requested budget."
        return result

    def _discovery_product(self, item: dict[str, Any], category: str | None, subcategory: str | None) -> Product:
        url = str(item.get("url", ""))
        return Product(
            product_id=f"{self.name}:{url}", source_product_id=url, source_name=self.name,
            name=item.get("title") or "Discovered product page", category=category or "",
            subcategory=subcategory or "", description=item.get("content") or item.get("snippet") or "",
            product_url=url, price=None, price_verified=False, availability="unknown",
            availability_verified=False, data_provenance={"name": {"source": url, "method": "tavily_search"}},
        )

    def get(self, product_id: str) -> Product | None:
        return None
