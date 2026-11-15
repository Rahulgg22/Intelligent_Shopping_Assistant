import pytest
from unittest.mock import MagicMock

from backend.providers.base import ProviderError
from backend.providers.composite import CompositeProductProvider
from backend.providers.serper import SerperShoppingProvider, parse_price_and_currency
from backend.schemas.product import Product, ProductSearchResult
from backend.services.product_retrieval import ProductRetrievalService
from backend.graph.workflow import shopping_graph
from backend.agents.session_memory import SESSIONS
from backend.mcp_server import search_products, get_provider_status, get_product_details
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(autouse=True)
def mock_external_ml_and_llm(monkeypatch):
    monkeypatch.setattr("backend.services.ai.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.intent_extraction.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.fallback.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.ranking._cross_encoder_scores", lambda query, candidates: None)


def test_serper_credentials_and_status():
    unconfigured = SerperShoppingProvider(api_key="")
    assert unconfigured.status()["state"] == "missing_credentials"
    with pytest.raises(ProviderError, match="not configured"):
        unconfigured.search("laptop")

    ready = SerperShoppingProvider(api_key="test-api-key")
    assert ready.status()["state"] == "ready"


def test_serper_http_errors_and_rate_limits(monkeypatch):
    class MockResponse:
        def __init__(self, status_code):
            self.status_code = status_code
        def raise_for_status(self):
            raise RuntimeError("unexpected")

    provider = SerperShoppingProvider(api_key="valid-key")

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: MockResponse(401))
    with pytest.raises(ProviderError, match="credentials were rejected"):
        provider.search("laptop")

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: MockResponse(429))
    with pytest.raises(ProviderError, match="rate limit reached"):
        provider.search("laptop")


def test_serper_currency_normalization_and_parsing():
    # INR formats
    val, curr, inr = parse_price_and_currency("₹54,990")
    assert val == 54990.0
    assert curr == "INR"
    assert inr == 54990.0

    val, curr, inr = parse_price_and_currency("Rs. 4,999.50")
    assert val == 4999.50
    assert curr == "INR"
    assert inr == 4999.50

    val, curr, inr = parse_price_and_currency("INR 60000")
    assert val == 60000.0
    assert inr == 60000.0

    # Foreign currency converted to INR
    val, curr, inr = parse_price_and_currency("$100")
    assert val == 100.0
    assert curr == "USD"
    assert inr == 8650.0  # 100 * 86.5

    # Missing / unverified price
    val, curr, inr = parse_price_and_currency("")
    assert inr is None

    val, curr, inr = parse_price_and_currency("Contact seller for price")
    assert inr is None


def test_serper_laptop_search_with_budget(monkeypatch):
    mock_payload = {
        "shopping": [
            {
                "title": "HP Pavilion 15 Core i5 Laptop",
                "source": "Amazon.in",
                "link": "https://www.amazon.in/dp/B0EXAMPL1",
                "price": "₹54,990",
                "delivery": "Free delivery",
                "rating": 4.3,
                "ratingCount": 210,
                "productId": "hp-pav-15",
            },
            {
                "title": "Lenovo ThinkPad E14 Programming Notebook",
                "source": "Lenovo Store",
                "link": "https://www.lenovo.com/in/thinkpad-e14",
                "price": "₹58,500",
                "delivery": "Ships in 2 days",
                "rating": 4.5,
                "ratingCount": 85,
                "productId": "lenovo-e14",
            },
            {
                "title": "Dell XPS 15 High Performance",
                "source": "Dell India",
                "link": "https://www.dell.com/xps15",
                "price": "₹1,25,000",
                "delivery": "Free delivery",
                "rating": 4.8,
                "ratingCount": 350,
                "productId": "dell-xps-15",
            },
            {
                "title": "Laptop Guide Review Article",
                "source": "TechGuide",
                "link": "https://techguide.example/best-laptops",
                "price": "",  # Unverified
                "productId": "guide-1",
            },
        ]
    }

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return mock_payload

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: MockResponse())
    provider = SerperShoppingProvider(api_key="test-key")
    result = provider.search("laptop for programming", max_price=60000)

    # Only products under ₹60,000 with verified prices survive
    assert len(result.products) == 2
    assert result.products[0].name == "HP Pavilion 15 Core i5 Laptop"
    assert result.products[0].price == 54990.0
    assert result.products[0].price_verified is True
    assert result.products[0].source_name == "Amazon.in"
    assert result.products[1].name == "Lenovo ThinkPad E14 Programming Notebook"
    assert result.products[1].price == 58500.0


def test_serper_cabin_suitcase_search(monkeypatch):
    mock_payload = {
        "shopping": [
            {
                "title": "American Tourister 55cm Cabin Trolley Suitcase",
                "source": "Flipkart",
                "link": "https://www.flipkart.com/at-cabin-trolley",
                "price": "₹4,299",
                "delivery": "Free delivery",
                "rating": 4.4,
                "ratingCount": 1200,
                "productId": "at-55-cabin",
            },
            {
                "title": "Samsonite Black Label Premium International Suitcase",
                "source": "Samsonite India",
                "link": "https://www.samsonite.in/black-label",
                "price": "₹28,000",
                "productId": "samsonite-premium",
            },
        ]
    }

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return mock_payload

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: MockResponse())
    provider = SerperShoppingProvider(api_key="test-key")
    result = provider.search("cabin suitcase", max_price=6000)

    assert len(result.products) == 1
    assert result.products[0].name.startswith("American Tourister")
    assert result.products[0].price == 4299.0
    assert result.products[0].price_verified is True


def test_serper_headphones_noise_cancellation(monkeypatch):
    mock_payload = {
        "shopping": [
            {
                "title": "boAt Nirvana 751 ANC Hybrid Active Noise Cancelling Headphones",
                "source": "boAt Lifestyle",
                "link": "https://www.boat-lifestyle.com/nirvana-751",
                "price": "₹3,499",
                "delivery": "Free shipping",
                "rating": 4.2,
                "ratingCount": 840,
                "productId": "boat-751",
            },
            {
                "title": "Sony WH-1000XM5 Wireless Noise Cancelling",
                "source": "Croma",
                "link": "https://www.croma.com/sony-xm5",
                "price": "₹29,990",
                "productId": "sony-xm5",
            },
        ]
    }

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return mock_payload

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: MockResponse())
    provider = SerperShoppingProvider(api_key="test-key")
    result = provider.search("wireless headphones noise cancellation", max_price=5000)

    assert len(result.products) == 1
    assert "boAt Nirvana" in result.products[0].name
    assert result.products[0].price == 3499.0
    assert result.products[0].price_verified is True


def test_serper_over_budget_failure_reporting(monkeypatch):
    mock_payload = {
        "shopping": [
            {"title": "Luxury Laptop", "source": "Store", "link": "https://example.com/lux", "price": "₹90,000", "productId": "lux-1"},
        ]
    }

    class MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return mock_payload

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: MockResponse())
    provider = SerperShoppingProvider(api_key="test-key")
    result = provider.search("laptop", max_price=50000)

    assert result.products == []
    assert result.provider_status["serper_shopping"] == "success_with_no_results"
    assert "exceeded the maximum budget" in result.errors["serper_shopping"]


def test_composite_preserves_multiple_retailer_offers():
    offer_amazon = Product(
        product_id="serper:amazon:model_1",
        name="Dell Inspiron 15",
        price=52000,
        currency="INR",
        price_verified=True,
        source_name="Amazon.in",
        source_product_id="model_1",
        product_url="https://www.amazon.in/dp/dell15",
        availability="in_stock",
    )
    offer_flipkart = Product(
        product_id="serper:flipkart:model_1",
        name="Dell Inspiron 15",
        price=53500,
        currency="INR",
        price_verified=True,
        source_name="Flipkart",
        source_product_id="model_1",
        product_url="https://www.flipkart.com/dell15",
        availability="in_stock",
    )

    class StaticProvider:
        name = "test_source"
        def status(self): return {"name": self.name, "state": "ready"}
        def search(self, *args, **kwargs):
            return ProductSearchResult(products=[offer_amazon, offer_flipkart], providers=[self.name])
        def get(self, product_id): return None

    composite = CompositeProductProvider([StaticProvider()])
    result = composite.search("dell inspiron")

    # Both retailer offers are preserved
    assert len(result.products) == 2
    merchants = {p.source_name for p in result.products}
    assert merchants == {"Amazon.in", "Flipkart"}
    prices = {p.price for p in result.products}
    assert prices == {52000, 53500}


def test_langgraph_workflow_with_mocked_shopping_service(monkeypatch):
    SESSIONS.clear()

    real_mock_laptop = Product(
        product_id="serper:amazon:dell_3520",
        name="Dell Inspiron 3520 Programming Laptop",
        category="laptops and electronics",
        subcategory="Laptops",
        price=48990,
        currency="INR",
        price_verified=True,
        source_name="Amazon.in",
        source_product_id="dell_3520",
        product_url="https://www.amazon.in/dp/dell3520",
        availability="in_stock",
        features=["16GB RAM", "512GB SSD", "Intel Core i5"],
        use_cases=["programming", "office"],
    )

    class MockShoppingProvider:
        name = "serper_shopping"
        def status(self): return {"name": self.name, "state": "ready"}
        def search(self, query="", max_price=None, **kwargs):
            if max_price is not None and real_mock_laptop.price > max_price:
                return ProductSearchResult(products=[], providers=[self.name], provider_status={self.name: "success_with_no_results"})
            return ProductSearchResult(products=[real_mock_laptop], providers=[self.name], provider_status={self.name: "success_with_results"})
        def get(self, pid): return real_mock_laptop if pid == real_mock_laptop.product_id else None

    import backend.graph.workflow as wf
    import backend.services.product_sources as sources_mod
    monkeypatch.setattr(sources_mod, "build_product_service", lambda: ProductRetrievalService(CompositeProductProvider([MockShoppingProvider()])))
    monkeypatch.setattr(wf, "build_product_service", lambda: ProductRetrievalService(CompositeProductProvider([MockShoppingProvider()])))

    # 1. Budget met
    res1 = wf.run_shopping_graph("serper_session_1", "Find a laptop under ₹60,000 for programming.")
    assert res1["action"] == "recommendations"
    assert len(res1["recommended_products"]) == 1
    assert res1["recommended_products"][0]["price"] == 48990
    assert res1["recommended_products"][0]["source_name"] == "Amazon.in"

    # 2. Strict budget lower than item price
    res2 = wf.run_shopping_graph("serper_session_2", "Find a laptop under ₹30,000 for programming.")
    assert res2["action"] == "message"
    assert len(res2.get("recommended_products", [])) == 0


def test_fastapi_chat_endpoint_contract(monkeypatch):
    client = TestClient(app)
    mock_product = {
        "product_id": "serper:croma:boat751",
        "name": "boAt Nirvana 751 ANC",
        "category": "laptops and electronics",
        "subcategory": "Headphones",
        "price": 3499.0,
        "currency": "INR",
        "price_verified": True,
        "in_stock": True,
        "product_url": "https://www.croma.com/boat751",
        "source_name": "Croma",
        "features": ["Active Noise Cancellation", "65H Playback"],
    }

    class MockRetrieval:
        def search(self, *args, **kwargs):
            return ProductSearchResult(
                products=[Product.model_validate(mock_product)],
                providers=["serper_shopping"],
                provider_status={"serper_shopping": "ready"},
            )
        def get(self, *args, **kwargs): return None
        def status(self): return {"name": "composite", "state": "ready"}

    monkeypatch.setattr("backend.graph.workflow.build_product_service", lambda: ProductRetrievalService(MockRetrieval()))

    response = client.post("/api/v1/chat", json={
        "session_id": "test-fastapi",
        "message": "Find wireless headphones under ₹5,000 with noise cancellation"
    })
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "action" in data
    assert "message" in data
    assert "products" in data
    assert len(data["products"]) == 1
    assert data["products"][0]["name"] == "boAt Nirvana 751 ANC"
    assert data["products"][0]["price"] == 3499.0
    assert data["products"][0]["price_verified"] is True


def test_mcp_tools_use_composite_shopping_service(monkeypatch):
    mock_item = Product(
        product_id="serper:flipkart:suit_1",
        name="Safari Thorium Cabin Luggage",
        price=2999,
        currency="INR",
        price_verified=True,
        source_name="Flipkart",
        product_url="https://www.flipkart.com/safari-thorium",
        availability="in_stock",
    )
    class MockService:
        def search(self, *args, **kwargs):
            return ProductSearchResult(products=[mock_item], providers=["serper_shopping"], provider_status={"serper_shopping": "ready"})
        def get(self, product_id):
            return mock_item if product_id == mock_item.product_id else None
        @property
        def provider(self):
            class ProvList:
                providers = [SerperShoppingProvider(api_key="key")]
            return ProvList()

    import backend.mcp_server as mcp_mod
    monkeypatch.setattr(mcp_mod, "build_product_service", lambda: MockService())

    search_res = search_products("cabin suitcase", max_price=6000)
    assert len(search_res["products"]) == 1
    assert search_res["products"][0]["name"] == "Safari Thorium Cabin Luggage"

    details_res = get_product_details("serper:flipkart:suit_1")
    assert details_res["product"] is not None
    assert details_res["product"]["price"] == 2999

    status_res = get_provider_status()
    assert "serper_shopping" in status_res["providers"]
