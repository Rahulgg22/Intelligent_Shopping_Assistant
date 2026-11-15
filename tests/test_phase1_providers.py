import pytest

from backend.providers.composite import CompositeProductProvider
from backend.providers.web_search import TavilyWebSearchProvider
from backend.providers.base import ProviderError
from backend.schemas.product import Product, ProductSearchResult


def make_product(product_id, *, source_name="test", source_product_id=None, price=1000):
    return Product(
        product_id=product_id,
        source_name=source_name,
        source_product_id=source_product_id or product_id,
        name="Demo Product",
        category="Electronics",
        price=price,
        product_url=f"https://{source_name}.example/products/{product_id}",
        features=["portable"],
        in_stock=True,
        availability="in_stock",
    )


class StaticProvider:
    def __init__(self, name, products=None, error=None):
        self.name = name
        self.products = products or []
        self.error = error

    def status(self):
        return {"name": self.name, "state": "ready"}

    def search(self, query="", **filters):
        if self.error:
            raise self.error
        products = [
            product for product in self.products
            if filters.get("max_price") is None or product.price <= filters["max_price"]
        ]
        return ProductSearchResult(products=products, providers=[self.name])

    def get(self, product_id):
        return next((product for product in self.products if product.product_id == product_id), None)


def test_product_normalizes_lists_booleans_and_missing_price():
    product = Product.from_record({
        "product_id": "x1",
        "name": "Example",
        "features": "lightweight; portable",
        "in_stock": "no",
        "price": "",
    })
    assert product.features == ["lightweight", "portable"]
    assert product.in_stock is False
    assert product.price is None
    assert product.availability == "out_of_stock"


def test_product_rejects_unsafe_urls():
    with pytest.raises(ValueError):
        Product(product_id="x1", product_url="http://localhost/private")
    with pytest.raises(ValueError):
        Product(product_id="x1", product_url="javascript:alert(1)")


def test_composite_isolates_provider_failure_and_reports_status():
    good = StaticProvider("good", [make_product("good-1")])
    bad = StaticProvider("bad", error=RuntimeError("temporarily unavailable"))
    result = CompositeProductProvider([good, bad]).search()
    assert [product.product_id for product in result.products] == ["good-1"]
    assert result.provider_status == {"good": "success_with_results", "bad": "error"}
    assert "bad" in result.errors


def test_composite_preserves_separate_retailer_offers():
    first = make_product("offer-1", source_name="store-a", source_product_id="model-1", price=1000)
    second = make_product("offer-2", source_name="store-b", source_product_id="model-1", price=1100)
    result = CompositeProductProvider([
        StaticProvider("store-a", [first]),
        StaticProvider("store-b", [second]),
    ]).search()
    assert len(result.products) == 2
    assert {product.source_name for product in result.products} == {"store-a", "store-b"}


def test_empty_composite_is_explicitly_unconfigured():
    result = CompositeProductProvider([]).search("laptop")
    assert result.products == []
    assert result.providers == []
    assert CompositeProductProvider([]).status()["state"] == "unconfigured"


def test_tavily_missing_credentials_is_explicit():
    provider = TavilyWebSearchProvider(api_key="")
    assert provider.status()["state"] == "missing_credentials"
    with pytest.raises(ProviderError):
        provider.search("laptop")


def test_tavily_normalizes_unverified_results(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        status_code = 200

        def json(self):
            return {"results": [{"title": "Laptop listing", "url": "https://store.example/laptop", "content": "Product page"}]}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response())
    provider = TavilyWebSearchProvider(api_key="test-key")
    monkeypatch.setattr(provider.extractor, "extract", lambda url, fallback=None: fallback)
    result = provider.search("laptop")
    assert len(result.products) == 1
    assert result.products[0].price is None
    assert result.products[0].price_verified is False