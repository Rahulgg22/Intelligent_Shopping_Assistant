import json

import pytest

from backend.providers.page_extractor import PageExtractor
from backend.providers.web_search import TavilyWebSearchProvider
from backend.providers.base import ProviderError


def test_tavily_auth_and_rate_limit_errors(monkeypatch):
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            raise RuntimeError("unexpected")

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response(401))
    with pytest.raises(ProviderError, match="rejected"):
        TavilyWebSearchProvider(api_key="key").search("laptop")
    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response(429))
    with pytest.raises(ProviderError, match="rate limit"):
        TavilyWebSearchProvider(api_key="key").search("laptop")


def test_page_extractor_parses_json_ld_and_provenance(monkeypatch):
    html = '<html><script type="application/ld+json">' + json.dumps({
        "@type": "Product", "name": "Example Laptop", "brand": {"name": "Example"},
        "sku": "MODEL-1", "offers": {"price": "54,990", "priceCurrency": "INR", "availability": "https://schema.org/InStock"},
    }) + "</script></html>"

    class Response:
        status_code = 200
        content = html.encode()
        headers = {"content-type": "text/html"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr("httpx.Client.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(PageExtractor, "_safe_url", lambda self, url: True)
    product = PageExtractor().extract("https://store.example/laptop")
    assert product.name == "Example Laptop"
    assert product.price == 54990
    assert product.price_verified is True
    assert product.availability == "in_stock"
    assert product.data_provenance["price"]["method"] == "json_ld_offer"


def test_page_extractor_rejects_unsafe_urls():
    assert PageExtractor().extract("http://127.0.0.1/admin") is None
    assert PageExtractor().extract("javascript:alert(1)") is None