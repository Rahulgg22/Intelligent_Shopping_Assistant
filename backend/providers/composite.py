from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, wait

from backend.providers.base import ProductProvider
from backend.schemas.product import Product, ProductSearchResult


class CompositeProductProvider:
    """Combine configured providers without allowing one failed source to block others."""

    name = "composite"

    def __init__(self, providers: Iterable[ProductProvider], deadline_seconds: float = 12.0, min_primary_results: int = 5):
        self.providers = list(providers)
        self.deadline_seconds = deadline_seconds
        self.min_primary_results = min_primary_results

    def status(self) -> dict[str, str]:
        return {"name": self.name, "state": "ready" if self.providers else "unconfigured"}

    def search(self, query: str = "", **filters) -> ProductSearchResult:
        products: dict[str, Product] = {}
        providers = []
        statuses = {}
        errors = {}
        primary = [provider for provider in self.providers if not getattr(provider, "discovery", False)]
        discovery = [provider for provider in self.providers if getattr(provider, "discovery", False)]

        self._run_group(primary, query, filters, products, providers, statuses, errors)
        if len(products) < self.min_primary_results:
            self._run_group(discovery, query, filters, products, providers, statuses, errors)
        else:
            for provider in discovery:
                statuses[provider.name] = "skipped_enough_results"
        return ProductSearchResult(
            products=list(products.values())[: filters.get("limit", 50)],
            providers=list(dict.fromkeys(providers)),
            errors=errors,
            provider_status=statuses,
        )

    def _run_group(self, providers, query, filters, products, provider_names, statuses, errors):
        if not providers:
            return
        pool = ThreadPoolExecutor(max_workers=len(providers), thread_name_prefix="product-provider")
        started = {provider.name: time.perf_counter() for provider in providers}
        futures = {
            pool.submit(provider.search, query, **filters): provider
            for provider in providers
        }
        done, pending = wait(futures, timeout=self.deadline_seconds)
        logger = logging.getLogger("providers")
        for future in pending:
            provider = futures[future]
            statuses[provider.name] = "timeout"
            errors[provider.name] = f"{provider.name} timed out after {self.deadline_seconds:g}s"
        for future in done:
            provider = futures[future]
            try:
                result = future.result()
                statuses[provider.name] = result.provider_status.get(provider.name, "success_with_results" if result.products else "success_with_no_results")
                provider_names.extend(result.providers or [provider.name])
                errors.update(result.errors)
                for product in result.products:
                    offer_key = (product.source_name, product.source_product_id or product.product_id, product.product_url)
                    products.setdefault(offer_key, product)
            except Exception as exc:
                statuses[provider.name] = "error"
                errors[provider.name] = str(exc)
            finally:
                logger.info("provider=%s took=%.2fs", provider.name, time.perf_counter() - started[provider.name])
        pool.shutdown(wait=False, cancel_futures=True)

    def get(self, product_id: str) -> Product | None:
        for provider in self.providers:
            try:
                product = provider.get(product_id)
            except Exception:
                continue
            if product is not None:
                return product
        return None