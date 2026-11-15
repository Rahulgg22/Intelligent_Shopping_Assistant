from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from backend.config import settings
from backend.providers.base import ProviderError
from backend.schemas.product import Product, ProductSearchResult

# Benchmark conversion rates to INR
EXCHANGE_RATES_TO_INR: dict[str, float] = {
    "INR": 1.0,
    "USD": 86.5,
    "EUR": 94.0,
    "GBP": 110.0,
    "AED": 23.5,
    "SGD": 65.0,
    "CAD": 63.0,
    "AUD": 56.5,
}

KNOWN_BRANDS = [
    "American Tourister", "Lattitude By Provogue", "Provogue", "Nasher Miles",
    "Tommy Hilfiger", "Samsonite", "Skybags", "Safari", "VIP", "Aristocrat",
    "Wildcraft", "Mokobara", "Uppercase", "Kamiliant", "Delsey", "Snitch",
    "Wilson", "Babolat", "Head", "Yonex", "Nike", "Adidas", "Puma",
    "Decathlon", "Artengo",
]


def extract_brand(title: str) -> str:
    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(brand)}\b", title, re.IGNORECASE):
            return brand
    return ""


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def parse_price_and_currency(raw_price: Any) -> tuple[float | None, str, float | None]:
    """Parse raw price string into (original_number, currency_code, inr_normalized_number)."""
    if raw_price is None or raw_price == "":
        return None, "INR", None

    text = str(raw_price).strip()
    lowered = text.lower()

    if "₹" in text or re.search(r"\brs\b\.?", lowered) or "inr" in lowered:
        currency = "INR"
    elif "$" in text or "usd" in lowered:
        currency = "USD"
    elif "€" in text or "eur" in lowered:
        currency = "EUR"
    elif "£" in text or "gbp" in lowered:
        currency = "GBP"
    elif "aed" in lowered:
        currency = "AED"
    elif "sgd" in lowered:
        currency = "SGD"
    elif "cad" in lowered:
        currency = "CAD"
    elif "aud" in lowered:
        currency = "AUD"
    else:
        currency = "INR"

    match = re.search(r"(\d[\d,]*(?:\.\d+)?)", text)
    if not match:
        return None, currency, None

    cleaned = match.group(1).replace(",", "")
    try:
        val = float(cleaned)
        rate = EXCHANGE_RATES_TO_INR.get(currency, 1.0)
        inr_val = round(val * rate, 2)
        return val, currency, inr_val
    except (ValueError, TypeError):
        return None, currency, None


class SerperShoppingProvider:
    """Google Shopping provider via Serper.dev API for real online merchant listings."""

    name = "serper_shopping"

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        gl: str | None = None,
        hl: str | None = None,
        timeout: float | None = None,
    ):
        self.api_key = (settings.serper_api_key if api_key is None else api_key) or ""
        self.endpoint = settings.serper_endpoint if endpoint is None else endpoint
        self.gl = settings.serper_gl if gl is None else gl
        self.hl = settings.serper_hl if hl is None else hl
        self.timeout = settings.serper_timeout_seconds if timeout is None else timeout

    def status(self) -> dict[str, str]:
        if not self.api_key:
            return {"name": self.name, "state": "missing_credentials"}
        return {"name": self.name, "state": "ready"}

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
        if not self.api_key:
            raise ProviderError("Serper API key is not configured")

        focused_parts = [part for part in (subcategory, query) if part]
        search_query = " ".join(dict.fromkeys(" ".join(focused_parts).split()))
        if not search_query:
            search_query = category or "products"

        payload = {
            "q": search_query,
            "gl": self.gl,
            "hl": self.hl,
            "num": min(max(limit, 1), settings.serper_num),
        }
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code in (401, 403):
                raise ProviderError("Serper credentials were rejected")
            if response.status_code == 429:
                raise ProviderError("Serper rate limit reached")
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderError("Serper request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Serper search failed: {exc}") from exc

        raw_items = data.get("shopping", [])
        if not raw_items:
            return ProductSearchResult(
                products=[],
                providers=[self.name],
                provider_status={self.name: "success_with_no_results"},
            )

        products: list[Product] = []
        raw_count = len(raw_items)
        unverified_count = 0
        over_budget_count = 0

        for item in raw_items:
            candidate = self._parse_item(item, category=category, subcategory=subcategory)
            if candidate is None:
                continue
            if re.search(r"\bset of \d|\b\d\s*(?:pcs?|pieces?)\b|\bcombo\b", candidate.name, re.IGNORECASE) and not re.search(r"\b(set|combo)\b", query, re.IGNORECASE):
                continue

            if candidate.price is None or not candidate.price_verified:
                unverified_count += 1
                if max_price is not None or min_price is not None:
                    continue
            if max_price is not None:
                if candidate.price > max_price:
                    over_budget_count += 1
                    continue
            if min_price is not None and candidate.price < min_price:
                continue

            products.append(candidate)

        has_products = len(products) > 0
        provider_status = {self.name: "success_with_results" if has_products else "success_with_no_results"}
        errors: dict[str, str] = {}

        if not has_products and raw_count > 0:
            if max_price is not None and over_budget_count > 0:
                errors[self.name] = (
                    f"Found {raw_count} real shopping listings, but all exceeded the maximum budget of ₹{max_price:g}."
                )
            elif max_price is not None and unverified_count == raw_count:
                errors[self.name] = (
                    f"Found {raw_count} shopping listings, but none had verifiable pricing for the requested budget."
                )
            else:
                errors[self.name] = "No listings matched the requested criteria."

        return ProductSearchResult(
            products=products[:limit],
            providers=[self.name],
            errors=errors,
            provider_status=provider_status,
        )

    def _parse_item(
        self,
        item: dict[str, Any],
        category: str | None = None,
        subcategory: str | None = None,
    ) -> Product | None:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        if not title or not link:
            return None

        # Validate URL scheme and safety
        parsed_url = urlparse(link)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            return None

        source_merchant = (item.get("source") or "Google Shopping").strip()
        product_id_str = str(item.get("productId") or stable_id(link))
        merchant_slug = re.sub(r"\W+", "_", source_merchant.lower()).strip("_")
        canonical_id = f"serper:{merchant_slug}:{product_id_str}"

        raw_val, raw_curr, inr_price = parse_price_and_currency(item.get("price"))
        price_verified = inr_price is not None and inr_price > 0

        rating_val = item.get("rating")
        rating: float | None = None
        if rating_val is not None:
            try:
                rating = float(rating_val)
            except (ValueError, TypeError):
                rating = None

        reviews_val = item.get("ratingCount")
        review_count: int | None = None
        if reviews_val is not None:
            try:
                review_count = int(reviews_val)
            except (ValueError, TypeError):
                review_count = None

        features: list[str] = []
        delivery = item.get("delivery")
        if delivery:
            features.append(str(delivery).strip())
        offers = item.get("offers")
        if offers:
            features.append(f"{offers} seller offers")

        description_parts = [f"Listed by {source_merchant} on Google Shopping."]
        if delivery:
            description_parts.append(f"Delivery: {delivery}.")
        if offers:
            description_parts.append(f"Alternative offers: {offers}.")

        return Product(
            product_id=canonical_id,
            name=title,
            category=category or "",
            subcategory=subcategory or "",
            brand=extract_brand(title),
            price=inr_price,
            currency="INR",
            original_price=None,
            description=" ".join(description_parts),
            features=features,
            rating=rating,
            review_count=review_count,
            image_url=item.get("imageUrl"),
            product_url=link,
            source_name=source_merchant,
            source_product_id=product_id_str,
            availability="unknown",
            price_verified=price_verified,
            availability_verified=False,
            retrieved_at=datetime.now(timezone.utc),
            data_provenance={
                "merchant": source_merchant,
                "raw_price": item.get("price"),
                "raw_price_value": raw_val,
                "raw_price_currency": raw_curr,
                "link_type": "google_shopping" if "google." in parsed_url.netloc else "merchant",
                "currency": raw_curr,
                "delivery": delivery,
                "google_product_id": item.get("productId"),
                "position": item.get("position"),
                "method": "serper_shopping_search",
            },
        )

    def get(self, product_id: str) -> Product | None:
        return None
