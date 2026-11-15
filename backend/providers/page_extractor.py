from __future__ import annotations

import ipaddress
import json
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from backend.schemas.product import Product


class PageExtractor:
    """Non-JavaScript JSON-LD/Open Graph extractor with SSRF defenses."""

    def __init__(self, timeout: float = 8.0, max_bytes: int = 1_500_000):
        self.timeout = timeout
        self.max_bytes = max_bytes

    def extract(self, url: str, fallback: Product | None = None) -> Product | None:
        if not self._safe_url(url):
            return fallback
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
                response = client.get(url, headers={"Accept": "text/html,application/xhtml+xml"})
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    target = urljoin(url, location or "")
                    if not location or not self._safe_url(target):
                        return fallback
                    response = client.get(target)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if content_type and "text/html" not in content_type and "application/xhtml" not in content_type:
                    return fallback
                content = response.content
                if len(content) > self.max_bytes:
                    content = content[: self.max_bytes]
        except (httpx.HTTPError, OSError, ValueError):
            return fallback
        return self._from_html(content, url, fallback)

    def _safe_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        host = parsed.hostname.lower().rstrip(".")
        if host == "localhost" or host.endswith(".local"):
            return False
        try:
            addresses = {info[4][0] for info in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)}
            return all(not self._unsafe_ip(address) for address in addresses)
        except OSError:
            return False

    @staticmethod
    def _unsafe_ip(value: str) -> bool:
        try:
            address = ipaddress.ip_address(value)
            return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved
        except ValueError:
            return True

    def _from_html(self, content: bytes, url: str, fallback: Product | None) -> Product | None:
        soup = BeautifulSoup(content, "html.parser")
        records = []
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                value = json.loads(script.string or script.get_text())
                records.extend(value if isinstance(value, list) else [value])
            except (json.JSONDecodeError, TypeError):
                continue
        record = next((item for item in records if isinstance(item, dict) and item.get("@type") == "Product"), None)
        if not record:
            return fallback
        offer = record.get("offers", {})
        if isinstance(offer, list):
            offer = offer[0] if offer else {}
        price = self._price(offer.get("price"))
        availability = str(offer.get("availability", "")).lower()
        stock = "out_of_stock" if "outofstock" in availability else "in_stock" if "instock" in availability else "unknown"
        image = record.get("image")
        image = image[0] if isinstance(image, list) and image else image
        brand = record.get("brand")
        brand = brand.get("name") if isinstance(brand, dict) else brand
        return Product(
            product_id=f"page:{url}", source_product_id=record.get("sku") or url,
            source_name=urlparse(url).netloc, name=record.get("name") or (fallback.name if fallback else "Product"),
            brand=brand or (fallback.brand if fallback else ""), description=record.get("description") or (fallback.description if fallback else ""),
            price=price, currency=offer.get("priceCurrency"), image_url=urljoin(url, image) if image else None,
            product_url=url, price_verified=price is not None, availability=stock,
            availability_verified=stock != "unknown", retrieved_at=datetime.now(timezone.utc),
            data_provenance={"price": {"source": url, "method": "json_ld_offer"}, "name": {"source": url, "method": "json_ld_product"}},
        )

    @staticmethod
    def _price(value: Any) -> float | None:
        try:
            number = float(str(value).replace(",", "").strip())
            return number if 0 < number < 100_000_000 else None
        except (TypeError, ValueError):
            return None
