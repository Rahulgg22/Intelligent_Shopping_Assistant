from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.alternatives import find_product_alternatives_impl
from backend.agents.comparison import compare_products_impl
from backend.agents.product_reviews import _classify_review, search_product_reviews_impl
from backend.agents.session_memory import SESSIONS, load_session, save_session
from backend.graph.workflow import run_shopping_graph
from backend.mcp_client import MCPClient, get_mcp_client
from backend.mcp_server import mcp
from backend.schemas.product import Product, ProductSearchResult


@pytest.fixture(autouse=True)
def reset_sessions_and_disable_encoder(monkeypatch):
    SESSIONS.clear()
    import backend.agents.ranking as ranking_mod
    monkeypatch.setattr(ranking_mod, "_cross_encoder_scores", lambda query, candidates: None)


# ============================================================================
# 1. compare_products Tests
# ============================================================================

def test_compare_two_products():
    products = [
        {
            "product_id": "p1",
            "name": "Asus Vivobook 15",
            "brand": "Asus",
            "price": 45000,
            "currency": "INR",
            "price_verified": True,
            "features": ["Intel Core i5", "8GB RAM", "512GB SSD", "FHD Display"],
            "rating": 4.2,
            "review_count": 120,
            "product_url": "https://example.com/p1",
        },
        {
            "product_id": "p2",
            "name": "Lenovo IdeaPad Slim 3",
            "brand": "Lenovo",
            "price": 52000,
            "currency": "INR",
            "price_verified": True,
            "features": ["Intel Core i5", "16GB RAM", "512GB SSD", "Backlit Keyboard"],
            "rating": 4.4,
            "review_count": 85,
            "product_url": "https://example.com/p2",
        },
    ]
    user_reqs = ["Suitable for programming", "Under ₹50,000", "16GB RAM"]

    result = compare_products_impl(products, user_reqs)

    assert len(result["comparison"]) == 2
    assert result["best_match"] is not None
    assert result["best_value"] is not None
    assert len(result["trade_offs"]) > 0
    assert "Price range: ₹45,000 to ₹52,000" in result["key_differences"]

    # Asus fulfills Under 50,000
    p1_eval = result["comparison"][0]
    assert any("Under ₹50,000" in req for req in p1_eval["matched_requirements"])

    # Lenovo fulfills 16GB RAM
    p2_eval = result["comparison"][1]
    assert any("16GB RAM" in req for req in p2_eval["matched_requirements"])


def test_compare_multiple_products():
    products = [
        {"title": "Laptop A", "price": 40000, "brand": "Brand A", "features": ["FHD Display", "512GB SSD"]},
        {"title": "Laptop B", "price": 50000, "brand": "Brand B", "features": ["FHD Display", "1TB SSD"]},
        {"title": "Laptop C", "price": 60000, "brand": "Brand C", "features": ["FHD Display", "16GB RAM"]},
    ]
    result = compare_products_impl(products)

    assert len(result["comparison"]) == 3
    assert "FHD Display" in result["common_features"]
    assert "Brands compared: Brand A, Brand B, Brand C" in result["key_differences"]
    assert result["best_match"] is not None


def test_compare_missing_specifications():
    products = [
        {
            "name": "Generic Office Laptop",
            "brand": "Generic",
            "price": 35000,
            "description": "Standard office laptop with nice keyboard.",
            "features": [],
            "specifications": {},
        },
        {
            "name": "Brand Laptop",
            "brand": "BrandX",
            "price": 38000,
            "features": ["8GB RAM", "Intel Core i3"],
        },
    ]
    result = compare_products_impl(products)

    p1_eval = result["comparison"][0]
    # Missing specs should be highlighted, not fabricated
    assert len(p1_eval["missing_specifications"]) > 0
    assert any("RAM / Memory" in m for m in p1_eval["missing_specifications"])
    assert any("Generic Office Laptop" in m for m in result["missing_information"])


def test_compare_unknown_prices():
    products = [
        {
            "name": "Unverified Laptop 1",
            "brand": "Alpha",
            "price": None,
            "price_verified": False,
            "features": ["Intel Core i7", "16GB RAM"],
            "rating": 4.5,
        },
        {
            "name": "Unverified Laptop 2",
            "brand": "Beta",
            "price": None,
            "price_verified": False,
            "features": ["Intel Core i5", "8GB RAM"],
            "rating": 4.0,
        },
    ]
    result = compare_products_impl(products, ["16GB RAM"])

    # Best value cannot be claimed without verified prices
    assert result["best_value"] is None
    assert any("verified prices are missing" in note.lower() for note in result["missing_information"])
    # Best match can still be evaluated based on specs
    assert result["best_match"] is not None
    assert result["best_match"]["name"] == "Unverified Laptop 1"


def test_best_match_and_best_value_distinction():
    products = [
        {
            "name": "Budget Champ",
            "brand": "Affordable",
            "price": 30000,
            "price_verified": True,
            "features": ["Intel Core i5", "8GB RAM", "programming"],
            "rating": 4.2,
        },
        {
            "name": "Pro Powerhouse",
            "brand": "Premium",
            "price": 85000,
            "price_verified": True,
            "features": ["Intel Core i9", "32GB RAM", "1TB SSD", "OLED Display", "programming", "fast compile"],
            "rating": 4.8,
        },
    ]
    # Requirements that both touch
    user_reqs = ["programming", "fast compile", "OLED Display"]

    result = compare_products_impl(products, user_reqs)

    # Pro Powerhouse satisfies more requirements -> Best match
    assert result["best_match"]["name"] == "Pro Powerhouse"
    # Budget Champ has far lower price with good rating -> Best value
    assert result["best_value"]["name"] == "Budget Champ"
    assert result["best_match"]["product_id"] != result["best_value"]["product_id"]


# ============================================================================
# 2. search_product_reviews Tests
# ============================================================================

def test_search_reviews_successful(monkeypatch):
    mock_tavily_resp = {
        "results": [
            {
                "title": "Sony WH-1000XM5 Review: The New King",
                "url": "https://www.theverge.com/2022/sony-wh-1000xm5-review",
                "content": "In our testing, we tested the headphones. Pros: exceptional noise cancellation, lightweight fit. Cons: expensive price tag.",
            },
            {
                "title": "My experience after 6 months with XM5",
                "url": "https://www.reddit.com/r/headphones/xm5_long_term",
                "content": "I bought this unit 6 months ago. Exceptional noise cancellation and fast performance, but expensive price tag.",
            },
        ]
    }

    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = mock_tavily_resp

    monkeypatch.setattr("httpx.post", mock_post)

    result = search_product_reviews_impl(
        product_name="Sony WH-1000XM5",
        brand="Sony",
        api_key="test-key",
    )

    assert result["review_status"] == "success"
    assert len(result["reviews"]) == 2
    assert result["reviews"][0]["review_type"] == "editorial"
    assert result["reviews"][1]["review_type"] == "customer"
    assert "theverge.com" in result["sources"]
    assert "reddit.com" in result["sources"]
    assert len(result["common_positive_points"]) > 0
    assert len(result["common_negative_points"]) > 0


def test_search_reviews_empty_results(monkeypatch):
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"results": []}
    monkeypatch.setattr("httpx.post", mock_post)

    result = search_product_reviews_impl(
        product_name="Obscure Rare Model 123",
        api_key="test-key",
    )

    assert result["review_status"] == "no_reviews_found"
    assert result["reviews"] == []


def test_search_reviews_duplicate_urls(monkeypatch):
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "results": [
            {"title": "Review 1", "url": "https://example.com/rev", "content": "Great sound"},
            {"title": "Review 1 Duplicate", "url": "https://example.com/rev", "content": "Great sound"},
        ]
    }
    monkeypatch.setattr("httpx.post", mock_post)

    result = search_product_reviews_impl(product_name="Sample Headphones", api_key="test-key")
    assert len(result["reviews"]) == 1


def test_search_reviews_provider_failure(monkeypatch):
    mock_post = MagicMock()
    mock_post.return_value.status_code = 429
    monkeypatch.setattr("httpx.post", mock_post)

    result = search_product_reviews_impl(product_name="Test Product", api_key="test-key")
    assert result["review_status"] == "provider_error"
    assert any("rate limit" in lim.lower() for lim in result["limitations"])


def test_editorial_vs_customer_classification():
    # Domain-based
    assert _classify_review("https://www.theverge.com/tech", "Title", "Body") == "editorial"
    assert _classify_review("https://rtings.com/headphones", "Title", "Body") == "editorial"
    assert _classify_review("https://reddit.com/r/laptops", "Title", "Body") == "customer"

    # Phrasing-based
    assert _classify_review("https://blog.xyz/post", "Review: In our testing", "We tested the unit") == "editorial"
    assert _classify_review("https://blog.xyz/post", "My daily driver", "I bought this laptop 3 months ago") == "customer"
    assert _classify_review("https://unknownsite.org/info", "Laptop Overview", "This is an overview") == "unknown"


# ============================================================================
# 3. find_product_alternatives Tests
# ============================================================================

def make_mock_product(pid: str, name: str, brand: str, price: float | None, verified: bool = True) -> Product:
    return Product(
        product_id=pid,
        name=name,
        brand=brand,
        price=price,
        currency="INR",
        price_verified=verified,
        category="Electronics",
        subcategory="Laptops",
        description=f"{name} by {brand}",
        product_url=f"https://example.com/{pid}",
    )


def test_alternatives_cheaper():
    orig_product = {
        "product_id": "orig_1",
        "name": "Dell XPS 13",
        "brand": "Dell",
        "price": 80000,
        "price_verified": True,
        "category": "Electronics",
    }
    candidates = [
        make_mock_product("alt_1", "Asus Zenbook 14", "Asus", 65000, True),
        make_mock_product("alt_2", "HP Envy 13", "HP", 72000, True),
        make_mock_product("alt_3", "MacBook Pro", "Apple", 120000, True),  # More expensive
        make_mock_product("alt_4", "Unknown Price Laptop", "Lenovo", None, False),  # Unverified
    ]

    mock_service = MagicMock()
    mock_service.search.return_value = ProductSearchResult(products=candidates, errors={})

    result = find_product_alternatives_impl(
        orig_product,
        alternative_type="cheaper",
        service=mock_service,
    )

    assert result["search_status"] == "success"
    alt_ids = [p["product_id"] for p in result["alternatives"]]
    assert "alt_1" in alt_ids
    assert "alt_2" in alt_ids
    assert "alt_3" not in alt_ids  # Rejected: strictly cheaper enforced
    assert "alt_4" not in alt_ids  # Rejected: unknown price cannot be verified cheaper


def test_alternatives_different_brands():
    orig_product = {
        "product_id": "orig_1",
        "name": "Dell Inspiron 15",
        "brand": "Dell",
        "price": 50000,
        "price_verified": True,
    }
    candidates = [
        make_mock_product("alt_1", "Dell Vostro 15", "Dell", 48000, True),  # Same brand
        make_mock_product("alt_2", "Lenovo IdeaPad 3", "Lenovo", 49000, True),
        make_mock_product("alt_3", "HP Pavilion 15", "HP", 51000, True),
    ]

    mock_service = MagicMock()
    mock_service.search.return_value = ProductSearchResult(products=candidates, errors={})

    result = find_product_alternatives_impl(
        orig_product,
        alternative_type="different_brands",
        service=mock_service,
    )

    assert result["search_status"] == "success"
    brands = [p["brand"].lower() for p in result["alternatives"]]
    assert "dell" not in brands
    assert "lenovo" in brands
    assert "hp" in brands


def test_alternatives_excluded_brands():
    orig_product = {"name": "Laptop", "brand": "BrandA", "price": 40000}
    candidates = [
        make_mock_product("alt_1", "Laptop B", "BrandB", 42000),
        make_mock_product("alt_2", "Laptop C", "ExcludedBrand", 41000),
    ]
    mock_service = MagicMock()
    mock_service.search.return_value = ProductSearchResult(products=candidates, errors={})

    result = find_product_alternatives_impl(
        orig_product,
        user_requirements={"excluded_brands": ["ExcludedBrand"]},
        service=mock_service,
    )
    brands = [p["brand"] for p in result["alternatives"]]
    assert "ExcludedBrand" not in brands
    assert "BrandB" in brands


def test_alternatives_budget_enforcement():
    orig_product = {"name": "Headphones", "price": 10000}
    candidates = [
        make_mock_product("alt_1", "Affordable Buds", "AudioX", 4500),
        make_mock_product("alt_2", "Pricier Buds", "AudioY", 7500),
    ]
    mock_service = MagicMock()
    mock_service.search.return_value = ProductSearchResult(products=candidates, errors={})

    result = find_product_alternatives_impl(
        orig_product,
        user_requirements={"max_price": 5000},
        service=mock_service,
    )
    assert len(result["alternatives"]) == 1
    assert result["alternatives"][0]["product_id"] == "alt_1"


def test_alternatives_no_results():
    orig_product = {"name": "Rare Laptop", "price": 50000}
    mock_service = MagicMock()
    mock_service.search.return_value = ProductSearchResult(products=[], errors={})

    result = find_product_alternatives_impl(orig_product, service=mock_service)
    assert result["search_status"] == "no_results"
    assert result["alternatives"] == []


def test_alternatives_provider_failure():
    orig_product = {"name": "Laptop"}
    mock_service = MagicMock()
    mock_service.search.side_effect = RuntimeError("Service unavailable")

    result = find_product_alternatives_impl(orig_product, service=mock_service)
    assert result["search_status"] == "provider_error"
    assert result["count"] == 0


# ============================================================================
# 4. MCP Server & Client Integration Tests
# ============================================================================

def test_mcp_server_has_all_registered_tools():
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    expected = [
        "search_products",
        "get_product_details",
        "get_provider_status",
        "compare_products",
        "search_product_reviews",
        "find_product_alternatives",
    ]
    for name in expected:
        assert name in tool_names, f"Tool '{name}' must be registered on MCP server"


def test_mcp_client_tool_discovery():
    client = get_mcp_client()
    tools = client.discover_tools()
    tool_map = {t["name"]: t for t in tools}

    assert "compare_products" in tool_map
    assert "search_product_reviews" in tool_map
    assert "find_product_alternatives" in tool_map

    # Check parameter schemas
    assert "products" in tool_map["compare_products"]["parameters"]["properties"]
    assert "product_name" in tool_map["search_product_reviews"]["parameters"]["properties"]
    assert "original_product" in tool_map["find_product_alternatives"]["parameters"]["properties"]


def test_mcp_client_invocation_of_compare_products():
    client = get_mcp_client()
    res = client.call_tool("compare_products", {
        "products": [
            {"name": "Option 1", "price": 20000, "brand": "Brand 1"},
            {"name": "Option 2", "price": 25000, "brand": "Brand 2"},
        ],
        "user_requirements": ["Under 22000"],
    })
    assert "comparison" in res
    assert len(res["comparison"]) == 2
    assert res["best_match"] is not None


# ============================================================================
# 5. LangGraph Planner Selection & Session Preservation Tests
# ============================================================================

def test_langgraph_turn1_search_then_turn2_compare(monkeypatch):
    """Verifies turn 1 produces products and turn 2 'Compare these three' uses MCP compare_products."""
    p1 = make_mock_product("p1", "HP Pavilion 15", "HP", 55000)
    p2 = make_mock_product("p2", "Dell Inspiron 15", "Dell", 58000)
    p3 = make_mock_product("p3", "Lenovo IdeaPad 3", "Lenovo", 52000)

    import backend.graph.workflow as wf_mod
    monkeypatch.setattr(
        wf_mod,
        "build_product_service",
        lambda: MagicMock(search=lambda *args, **kwargs: ProductSearchResult(products=[p1, p2, p3], errors={}))
    )

    # Turn 1: Search
    session_id = "test_compare_flow"
    res1 = run_shopping_graph(session_id, "Suggest laptops under ₹60,000 for programming")

    assert res1["action"] == "recommendations"
    assert len(res1["recommended_products"]) == 3

    # Verify session memory preserved the products
    saved_session = load_session({"session_id": session_id})
    assert len(saved_session["previous_products"]) == 3

    # Turn 2: Compare
    res2 = run_shopping_graph(session_id, "Compare the top three laptops")

    assert res2["action"] == "recommendations"
    assert res2["selected_tool"] == "compare_products"
    assert "comparison_result" in res2
    assert "side-by-side comparison" in res2["response_message"].lower()


def test_langgraph_turn2_reviews(monkeypatch):
    """Verifies turn 2 'What are the reviews like for this laptop?' invokes search_product_reviews."""
    p1 = make_mock_product("p1", "Sony WH-1000XM4", "Sony", 22000)

    import backend.graph.workflow as wf_mod
    monkeypatch.setattr(
        wf_mod,
        "build_product_service",
        lambda: MagicMock(search=lambda *args, **kwargs: ProductSearchResult(products=[p1], errors={}))
    )

    # Mock Tavily response for reviews
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "results": [
            {
                "title": "Sony WH-1000XM4 Review",
                "url": "https://www.soundguys.com/sony-xm4",
                "content": "Pros: excellent active noise cancellation, lightweight fit. Cons: microphone is mediocre.",
            }
        ]
    }
    monkeypatch.setattr("httpx.post", mock_post)

    session_id = "test_reviews_flow"
    run_shopping_graph(session_id, "Find wireless headphones under ₹25,000 with noise cancellation")

    # Turn 2: Ask reviews
    res2 = run_shopping_graph(session_id, "What are the reviews like for this headphone?")

    assert res2["selected_tool"] == "search_product_reviews"
    assert "review_result" in res2
    assert "What Reviewers Liked" in res2["response_message"]


def test_langgraph_turn2_alternatives(monkeypatch):
    """Verifies turn 2 'Suggest cheaper alternatives' invokes find_product_alternatives."""
    p1 = make_mock_product("p1", "Dell XPS 13", "Dell", 80000)
    alt1 = make_mock_product("alt1", "Asus Zenbook 14", "Asus", 65000)

    mock_service = MagicMock()
    # Turn 1 returns p1, Turn 2 returns alt1
    mock_service.search.side_effect = [
        ProductSearchResult(products=[p1], errors={}),
        ProductSearchResult(products=[alt1], errors={}),
    ]

    import backend.graph.workflow as wf_mod
    import backend.agents.alternatives as alt_mod
    monkeypatch.setattr(wf_mod, "build_product_service", lambda: mock_service)
    monkeypatch.setattr(alt_mod, "build_product_service", lambda: mock_service)

    session_id = "test_alt_flow"
    run_shopping_graph(session_id, "Suggest high end laptops for coding")

    # Turn 2: Ask cheaper alternatives
    res2 = run_shopping_graph(session_id, "Suggest cheaper alternatives to this laptop")

    assert res2["selected_tool"] == "find_product_alternatives"
    assert res2["action"] == "alternatives"
    assert len(res2["recommended_products"]) == 1
    assert res2["recommended_products"][0]["name"] == "Asus Zenbook 14"


def test_clarification_on_empty_context():
    """If user asks to compare or reviews in turn 1 with NO prior products, assistant asks clarification."""
    session_id = "test_empty_clarify"
    res = run_shopping_graph(session_id, "Compare these three laptops")
    assert res["action"] == "clarification"
    assert "search for products first" in res["response_message"].lower()

    session_id_2 = "test_empty_reviews"
    res2 = run_shopping_graph(session_id_2, "What are the reviews like?")
    assert res2["action"] == "clarification"
    assert "which product" in res2["response_message"].lower()


def test_reviews_single_source_lacks_multi_source_corroboration(monkeypatch):
    """Ensure points appearing in only one source are not marked as recurring consensus."""
    mock_tavily_resp = {
        "results": [
            {
                "title": "Review 1",
                "url": "https://www.theverge.com/rev1",
                "content": "In our testing. Cons: expensive price tag.",
            },
            {
                "title": "Review 2",
                "url": "https://www.reddit.com/r/laptops/rev2",
                "content": "My experience. Cons: fan noise is noticeable under load.",
            },
        ]
    }
    mock_post = MagicMock(return_value=MagicMock(status_code=200, json=lambda: mock_tavily_resp))
    monkeypatch.setattr("httpx.post", mock_post)

    res = search_product_reviews_impl("Laptop", api_key="k")
    # Neither con appeared in >=2 sources
    assert res["common_negative_points"] == []
    assert any("no recurring negative consensus" in lim.lower() for lim in res["limitations"])


def test_alternatives_better_specifications():
    """Ensure better_specifications requires actual extracted spec improvements, not search query alone."""
    orig = {
        "name": "Base Laptop 8GB",
        "description": "Intel Core i5, 8GB RAM, 512GB SSD",
        "price": 50000,
        "rating": 4.0,
    }
    # Candidate 1 has upgraded RAM (16GB) and CPU tier (Core i7) -> Qualified
    cand1 = make_mock_product("alt_upgrade", "Pro Laptop 16GB", "BrandX", 65000)
    cand1.features = ["Intel Core i7", "16GB RAM", "1TB SSD"]
    cand1.rating = 4.5

    # Candidate 2 has same or worse specs (4GB RAM) -> Rejected
    cand2 = make_mock_product("alt_downgrade", "Budget Laptop 4GB", "BrandY", 40000)
    cand2.features = ["Intel Core i3", "4GB RAM", "256GB SSD"]
    cand2.rating = 3.8

    mock_service = MagicMock()
    mock_service.search.return_value = ProductSearchResult(products=[cand1, cand2], errors={})

    res = find_product_alternatives_impl(orig, alternative_type="better_specifications", service=mock_service)
    alt_ids = [p["product_id"] for p in res["alternatives"]]
    assert "alt_upgrade" in alt_ids
    assert "alt_downgrade" not in alt_ids


def test_compare_incompatible_currencies():
    """Ensure products with unmapped/incompatible currencies are flagged and direct comparison is prevented."""
    products = [
        {"name": "Laptop INR", "price": 50000, "currency": "INR", "price_verified": True},
        {"name": "Laptop XYZ", "price": 400, "currency": "XYZ_UNMAPPED", "price_verified": True},
    ]
    res = compare_products_impl(products)
    assert res["best_value"] is None
    assert any("incompatible currencies" in diff.lower() for diff in res["key_differences"])


def test_langgraph_mcp_tool_error_graceful_recovery(monkeypatch):
    """Ensure unexpected MCP tool errors are caught and do not crash the LangGraph workflow."""
    import backend.graph.workflow as wf_mod
    client = get_mcp_client()

    # Force call_tool to raise an unexpected runtime error
    monkeypatch.setattr(client, "call_tool", MagicMock(side_effect=RuntimeError("Simulated MCP tool crash")))
    monkeypatch.setattr(wf_mod, "get_mcp_client", lambda: client)

    session_id = "test_tool_error_handling"
    # Seed session with previous products
    save_session({
        "session_id": session_id,
        "recommended_products": [
            {"name": "Laptop 1", "price": 50000},
            {"name": "Laptop 2", "price": 60000},
        ],
        "history": [],
        "constraints": {},
        "turn": 1,
    })

    # Call tool intent that encounters error
    res = run_shopping_graph(session_id, "Compare these laptops")
    assert res["action"] == "message"
    assert "issue" in res["response_message"].lower()
    assert res["selected_tool"] == "compare_error"
