import pytest
from unittest.mock import MagicMock

from backend.agents.ranking import (
    rank_candidates,
    get_ranking_weights,
    _deduplicate_candidates,
)
from backend.agents.explanation import explain_recommendations
from backend.agents.fallback import handle_fallback, diagnose_fallback_condition
from backend.agents.intent_extraction import extract_intent
from backend.agents.session_memory import SESSIONS, load_session, save_session
from backend.graph.workflow import shopping_graph, run_shopping_graph
from backend.schemas.product import Product, ProductSearchResult
from backend.services.product_retrieval import ProductRetrievalService


@pytest.fixture(autouse=True)
def mock_external_ml_and_llm(monkeypatch):
    monkeypatch.setattr("backend.services.ai.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.intent_extraction.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.clarification.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.explanation.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.fallback.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.ranking._cross_encoder_scores", lambda query, candidates: None)


# ==============================================================================
# 1. Explainable Product Ranking Tests
# ==============================================================================

def test_configurable_ranking_weights(monkeypatch):
    monkeypatch.setenv("WEIGHT_SEMANTIC", "0.35")
    monkeypatch.setenv("WEIGHT_LEXICAL", "0.15")
    monkeypatch.setenv("WEIGHT_ATTRIBUTE", "0.25")
    monkeypatch.setenv("WEIGHT_BUDGET", "0.15")
    monkeypatch.setenv("WEIGHT_BRAND", "0.10")

    weights = get_ranking_weights()
    assert weights["semantic"] == 0.35
    assert weights["lexical"] == 0.15
    assert weights["attribute"] == 0.25
    assert weights["budget"] == 0.15
    assert weights["brand"] == 0.10


def test_hard_budget_filtering_and_missing_prices():
    p_valid = {
        "product_id": "p1", "name": "Dell Inspiron 15 Laptop", "price": 45000,
        "brand": "Dell", "category": "Electronics", "price_verified": True
    }
    p_over = {
        "product_id": "p2", "name": "Dell XPS 15 Laptop", "price": 75000,
        "brand": "Dell", "category": "Electronics", "price_verified": True
    }
    p_no_price = {
        "product_id": "p3", "name": "Dell Latitude Laptop", "price": None,
        "brand": "Dell", "category": "Electronics", "price_verified": False
    }

    state = {
        "normalized_query": "laptop under 50000",
        "constraints": {"product_type": "laptop", "budget_max": 50000.0},
        "candidates": [p_valid, p_over, p_no_price],
    }

    result = rank_candidates(state)
    ranked = result["ranked_products"]
    rejections = result["rejection_reasons"]

    # Only p_valid should remain
    assert len(ranked) == 1
    assert ranked[0]["product_id"] == "p1"
    assert ranked[0]["rank"] == 1
    assert rejections["budget_exceeded"] == 1
    assert rejections["unverified_price"] == 1


def test_preferred_and_excluded_brands():
    p_apple = {"product_id": "p_app", "name": "Apple MacBook Air", "price": 70000, "brand": "Apple"}
    p_hp = {"product_id": "p_hp", "name": "HP Pavilion Laptop", "price": 55000, "brand": "HP"}
    p_lenovo = {"product_id": "p_len", "name": "Lenovo IdeaPad", "price": 52000, "brand": "Lenovo"}

    # Exclude Apple, prefer HP
    state = {
        "normalized_query": "laptop",
        "constraints": {
            "product_type": "laptop",
            "excluded_brands": ["apple"],
            "preferred_brands": ["hp"],
        },
        "candidates": [p_apple, p_hp, p_lenovo],
    }

    result = rank_candidates(state)
    ranked = result["ranked_products"]
    ids = [p["product_id"] for p in ranked]

    # Apple must be completely excluded
    assert "p_app" not in ids
    # HP must be ranked higher than Lenovo due to preferred brand boost
    assert ids[0] == "p_hp"
    assert any("HP" in req for req in ranked[0]["matched_requirements"])


def test_brand_diversity_in_ranking():
    # 4 products from same brand vs 1 from another brand
    cands = [
        {"product_id": "b1", "name": "boAt Rockerz 450", "price": 1499, "brand": "boAt", "features": ["wireless", "anc"]},
        {"product_id": "b2", "name": "boAt Rockerz 550", "price": 1799, "brand": "boAt", "features": ["wireless", "anc"]},
        {"product_id": "b3", "name": "boAt Rockerz 650", "price": 1999, "brand": "boAt", "features": ["wireless", "anc"]},
        {"product_id": "s1", "name": "Sony WH-CH520", "price": 2499, "brand": "Sony", "features": ["wireless"]},
    ]

    state = {
        "normalized_query": "wireless headphones",
        "constraints": {"product_type": "headphones", "brand_diversity": True},
        "candidates": cands,
    }

    result = rank_candidates(state)
    ranked = result["ranked_products"]
    top_brands = [p["brand"].lower() for p in ranked[:2]]
    # Top 2 results should include both brands, not just boAt
    assert "sony" in top_brands
    assert "boat" in top_brands


def test_candidate_deduplication():
    cands = [
        # Duplicate URL from multiple providers
        {"product_id": "1", "name": "boAt 751", "product_url": "https://amazon.in/dp/b751", "source_name": "Amazon", "price": 3499},
        {"product_id": "2", "name": "boAt 751 ANC", "product_url": "https://amazon.in/dp/b751", "source_name": "Amazon", "price": 3499},
        # Separate retailer offer for same item -> must be preserved
        {"product_id": "3", "name": "boAt 751", "product_url": "https://flipkart.com/dp/b751", "source_name": "Flipkart", "price": 3299},
    ]

    deduped = _deduplicate_candidates(cands)
    assert len(deduped) == 2
    merchants = {p["source_name"] for p in deduped}
    assert merchants == {"Amazon", "Flipkart"}


def test_structured_ranking_explanations():
    product = {
        "product_id": "lap_1",
        "name": "Dell Inspiron 3520 Programming Laptop",
        "category": "Electronics",
        "subcategory": "Laptops",
        "brand": "Dell",
        "price": 48990.0,
        "price_verified": True,
        "availability": "in_stock",
        "rating": 4.3,
        "features": ["16GB RAM", "512GB SSD", "Intel Core i5"],
        "use_cases": ["programming", "office"],
    }

    state = {
        "normalized_query": "laptop under 50000 for programming",
        "constraints": {
            "product_type": "laptop",
            "intended_use": "programming",
            "budget_max": 50000.0,
            "preferences": ["16gb ram"],
        },
        "ranked_products": [product],
    }

    res = explain_recommendations(state)
    assert res["action"] == "recommendations"
    assert res["search_status"] == "success"
    assert len(res["recommended_products"]) == 1

    item = res["recommended_products"][0]
    assert item["rank"] == 1
    assert "Laptop" in item["matched_requirements"]
    assert "For Programming" in item["matched_requirements"]
    assert any("16GB RAM" in req for req in item["matched_requirements"])
    assert len(item["benefits"]) > 0
    assert "programming" in item["explanation"].lower()
    assert "comparison_note" in item


def test_deterministic_ranking_stability():
    # Identical candidate inputs must yield identical ranking order
    cands = [
        {"product_id": "a", "name": "Asus Vivobook", "price": 42000, "brand": "Asus", "rating": 4.1},
        {"product_id": "b", "name": "Acer Aspire 5", "price": 44000, "brand": "Acer", "rating": 4.2},
        {"product_id": "c", "name": "Lenovo IdeaPad Slim 3", "price": 41000, "brand": "Lenovo", "rating": 4.0},
    ]
    state = {
        "normalized_query": "laptop under 45000",
        "constraints": {"product_type": "laptop", "budget_max": 45000.0},
        "candidates": cands,
    }

    res1 = rank_candidates(state)["ranked_products"]
    res2 = rank_candidates(state)["ranked_products"]

    ids1 = [p["product_id"] for p in res1]
    ids2 = [p["product_id"] for p in res2]
    assert ids1 == ids2


# ==============================================================================
# 2. Intelligent Fallback and Recovery Tests
# ==============================================================================

def test_fallback_provider_error():
    state = {
        "normalized_query": "laptop under 50000",
        "constraints": {"product_type": "laptop", "budget_max": 50000.0},
        "candidates": [],
        "retrieval_errors": {"serper_shopping": "HTTP 429: rate limit reached"},
        "provider_status": {"serper_shopping": "rate_limited"},
    }

    res = handle_fallback(state)
    assert res["action"] == "message"
    assert res["search_status"] == "provider_error"
    assert res["fallback_reason"] == "PROVIDER_ERROR"
    assert "retry" in res["response_message"].lower()
    assert "Retry search" in res["fallback_suggestions"]


def test_fallback_empty_retrieval_response():
    state = {
        "normalized_query": "xyz123randomnonexistentproduct",
        "constraints": {"product_type": "widget"},
        "candidates": [],
        "retrieval_errors": {},
        "provider_status": {"serper_shopping": "ready"},
    }

    res = handle_fallback(state)
    assert res["search_status"] == "no_results"
    assert res["fallback_reason"] == "NO_PRODUCTS_RETRIEVED"
    assert "Broaden search" in res["fallback_suggestions"]


def test_fallback_budget_restriction_with_lowest_observed_price():
    # Candidates existed, but cheapest laptop is ₹24,990 while user's budget is ₹15,000
    state = {
        "normalized_query": "laptop under 15000",
        "constraints": {"product_type": "laptop", "budget_max": 15000.0},
        "candidates": [
            {"product_id": "l1", "name": "Basic Laptop", "price": 24990.0},
            {"product_id": "l2", "name": "Medium Laptop", "price": 32000.0},
        ],
        "rejection_reasons": {
            "budget_exceeded": 2,
            "lowest_seen_price": 24990.0,
        },
    }

    res = handle_fallback(state)
    assert res["search_status"] == "no_results"
    assert res["fallback_reason"] == "BUDGET_RESTRICTION"
    assert "15,000" in res["response_message"]
    assert "24,990" in res["response_message"]
    assert any("25,000" in sug or "24,990" in sug for sug in res["fallback_suggestions"])


def test_fallback_brand_restriction():
    state = {
        "normalized_query": "shoes",
        "constraints": {
            "product_type": "shoes",
            "excluded_brands": ["nike", "adidas", "puma"],
        },
        "candidates": [
            {"product_id": "s1", "name": "Nike Air Zoom", "brand": "Nike", "price": 4000},
            {"product_id": "s2", "name": "Adidas Ultraboost", "brand": "Adidas", "price": 5000},
        ],
        "rejection_reasons": {
            "brand_excluded": 2,
            "lowest_seen_price": 4000,
        },
    }

    res = handle_fallback(state)
    assert res["fallback_reason"] == "BRAND_RESTRICTION"
    assert "brand" in res["response_message"].lower()
    assert "Include other brands" in res["fallback_suggestions"]


def test_fallback_specification_restriction():
    state = {
        "normalized_query": "laptop with 64gb ram and oled",
        "constraints": {
            "product_type": "laptop",
            "preferences": ["64gb ram", "oled 4k"],
        },
        "candidates": [
            {"product_id": "l1", "name": "Standard Laptop", "features": ["16GB RAM", "IPS"]},
        ],
        "rejection_reasons": {
            "specification_unmatched": 1,
            "lowest_seen_price": 45000,
        },
    }

    res = handle_fallback(state)
    assert res["fallback_reason"] == "SPECIFICATION_RESTRICTION"
    assert "Relax specifications" in res["fallback_suggestions"]


def test_fallback_unverified_prices():
    state = {
        "normalized_query": "rare collectable watch under 5000",
        "constraints": {
            "product_type": "watch",
            "budget_max": 5000.0,
        },
        "candidates": [
            {"product_id": "w1", "name": "Vintage Watch", "price": None, "price_verified": False},
        ],
        "rejection_reasons": {
            "unverified_price": 1,
            "lowest_seen_price": float("inf"),
        },
    }

    res = handle_fallback(state)
    assert res["search_status"] == "no_verified_results"
    assert res["fallback_reason"] == "UNVERIFIED_INFORMATION"
    assert "Search other retailers" in res["fallback_suggestions"]


def test_user_accepting_fallback_suggestion_preserves_context():
    # User originally searched: "Find a laptop for programming under ₹15,000"
    prev_constraints = {
        "product_type": "laptop",
        "category": "Electronics",
        "subcategory": "Laptops",
        "intended_use": "programming",
        "budget_max": 15000.0,
        "preferences": ["16gb ram"],
    }

    # User clicks fallback chip: "Increase budget to ₹25,000"
    followup_state = {
        "message": "Increase budget to ₹25,000",
        "constraints": prev_constraints,
    }

    extracted = extract_intent(followup_state)
    c = extracted["constraints"]

    # Budget is updated
    assert c["budget_max"] == 25000.0
    # Core context is preserved!
    assert c["product_type"] == "laptop"
    assert c["intended_use"] == "programming"
    assert "16gb ram" in c["preferences"]
    assert extracted["query_is_new"] is False


def test_user_accepting_include_other_brands_suggestion():
    prev_constraints = {
        "product_type": "shoes",
        "category": "Fashion and Clothing",
        "excluded_brands": ["nike"],
        "budget_max": 3000.0,
    }

    followup_state = {
        "message": "Include other brands",
        "constraints": prev_constraints,
    }

    extracted = extract_intent(followup_state)
    c = extracted["constraints"]

    assert c["excluded_brands"] == []
    assert c["brand_diversity"] is True
    assert c["product_type"] == "shoes"
    assert c["budget_max"] == 3000.0


def test_user_accepting_relax_specifications_suggestion():
    prev_constraints = {
        "product_type": "laptop",
        "budget_max": 60000.0,
        "preferences": ["64gb ram", "oled 4k"],
    }

    followup_state = {
        "message": "Relax specifications",
        "constraints": prev_constraints,
    }

    extracted = extract_intent(followup_state)
    c = extracted["constraints"]

    assert c["preferences"] == []
    assert c["product_type"] == "laptop"
    assert c["budget_max"] == 60000.0


# ==============================================================================
# 3. End-to-End Workflow Integration Test
# ==============================================================================

def test_full_workflow_from_budget_fallback_to_successful_recovery(monkeypatch):
    SESSIONS.clear()
    session_id = "test_budget_recovery_session"

    laptop_item = Product(
        product_id="serper:dell:inspiron",
        name="Dell Inspiron 3520",
        price=24990.0,
        currency="INR",
        price_verified=True,
        source_name="Amazon.in",
        product_url="https://amazon.in/dp/dell3520",
        availability="in_stock",
        features=["8GB RAM", "Intel Core i3"],
        use_cases=["office", "programming"],
    )

    class MockService:
        def search(self, query="", max_price=None, **kwargs):
            if max_price is not None and laptop_item.price > max_price:
                # Returns candidate that exceeds budget
                return ProductSearchResult(products=[laptop_item], providers=["serper_shopping"], provider_status={"serper_shopping": "ready"})
            return ProductSearchResult(products=[laptop_item], providers=["serper_shopping"], provider_status={"serper_shopping": "ready"})
        def get(self, pid): return laptop_item if pid == laptop_item.product_id else None

    import backend.graph.workflow as wf
    import backend.services.product_sources as sources_mod
    monkeypatch.setattr(sources_mod, "build_product_service", lambda: ProductRetrievalService(MockService()))
    monkeypatch.setattr(wf, "build_product_service", lambda: ProductRetrievalService(MockService()))

    # Turn 1: Budget too low (₹15,000)
    out1 = run_shopping_graph(session_id, "Find a laptop for programming under ₹15,000")
    assert out1["action"] == "message"
    assert out1["search_status"] == "no_results"
    assert out1["fallback_reason"] == "BUDGET_RESTRICTION"
    assert any("25,000" in sug or "24,990" in sug for sug in out1["fallback_suggestions"])

    # Turn 2: User accepts suggestion chip "Increase budget to ₹25,000"
    out2 = run_shopping_graph(session_id, "Increase budget to ₹25,000")
    assert out2["action"] == "recommendations"
    assert out2["search_status"] == "success"
    assert len(out2["recommended_products"]) == 1
    assert out2["recommended_products"][0]["name"] == "Dell Inspiron 3520"
    assert out2["recommended_products"][0]["rank"] == 1
    assert "comparison_note" in out2["recommended_products"][0]
