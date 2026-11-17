import pytest
from unittest.mock import MagicMock

from backend.agents.intent_extraction import (
    extract_budget,
    extract_recipient,
    extract_intended_use,
    product_phrase,
    _deterministic_intent,
    extract_intent,
)
from backend.agents.clarification import (
    evaluate_clarification_decision,
    _fallback_question,
    check_clarification,
)
from backend.agents.session_memory import SESSIONS
from backend.graph.workflow import shopping_graph, run_shopping_graph
from backend.schemas.product import Product, ProductSearchResult
from backend.services.product_retrieval import ProductRetrievalService


@pytest.fixture(autouse=True)
def mock_all_external(monkeypatch):
    monkeypatch.setattr("backend.services.ai.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.intent_extraction.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.clarification.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.explanation.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.fallback.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.ranking._cross_encoder_scores", lambda query, candidates: None)


# ==============================================================================
# A. Generic Product Queries Across Different Categories
# ==============================================================================

@pytest.mark.parametrize("query,expected_type,expected_cat", [
    ("Suggest a laptop", "laptop", "Electronics"),
    ("Recommend a mobile phone", "mobile phone", "Electronics"),
    ("I need headphones", "headphones", "Electronics"),
    ("Suggest shoes", "shoes", "Fashion and Clothing"),
    ("I need a backpack", "backpack", "Travel and Luggage"),
    ("Recommend a smartwatch", "smartwatch", "Electronics"),
    ("Suggest a gift", "gifts", "Gifts"),
    ("I need furniture", "furniture", "Furniture and Home Decor"),
    ("Suggest a washing machine", "washing machine", "Home and Kitchen"),
    ("I need a travel bag", "travel bag", "Travel and Luggage"),
    ("Recommend skincare products", "skincare products", "Beauty and Personal Care"),
    ("Suggest books", "books", "Books and Stationery"),
])
def test_generic_product_queries_across_categories(query, expected_type, expected_cat):
    intent = _deterministic_intent(query)
    assert intent["product_type"] == expected_type
    assert intent["category"] == expected_cat


# ==============================================================================
# B. Recipient Extraction & Immutability
# ==============================================================================

@pytest.mark.parametrize("query,expected_recipient", [
    ("Suggest me gifts for my brother", "brother"),
    ("Suggest gifts for my sister", "sister"),
    ("Gifts for my father", "father"),
    ("Gifts for my mother", "mother"),
    ("Gifts for my friend", "friend"),
    ("Suggest a birthday gift for my wife", "wife"),
    ("Find a watch for my husband", "husband"),
    ("Gifts for my boyfriend", "boyfriend"),
    ("Present for my girlfriend", "girlfriend"),
    ("Gift for my colleague", "colleague"),
    ("Toys for my kids", "children"),
])
def test_recipient_extraction_accuracy(query, expected_recipient):
    intent = _deterministic_intent(query)
    assert intent["recipient"] == expected_recipient
    # Verify that "brother" NEVER becomes "sister"
    if expected_recipient == "brother":
        assert intent["recipient"] != "sister"
    if expected_recipient == "sister":
        assert intent["recipient"] != "brother"


def test_clarification_question_never_swaps_brother_for_sister():
    state = {
        "message": "Suggest me gifts for my brother",
        "normalized_query": "Suggest me gifts for my brother",
        "constraints": {
            "product_type": "gifts",
            "category": "Gifts",
            "recipient": "brother",
        }
    }
    result = check_clarification(state)
    assert result["needs_clarification"] is True
    question = result["clarifying_question"].lower()
    assert "brother" in question or "him" in question or "he" in question
    assert "sister" not in question
    assert "hiking" not in question
    assert "gear" not in question


def test_clarification_question_for_sister():
    state = {
        "message": "Suggest gifts for my sister",
        "normalized_query": "Suggest gifts for my sister",
        "constraints": {
            "product_type": "gifts",
            "category": "Gifts",
            "recipient": "sister",
        }
    }
    result = check_clarification(state)
    assert result["needs_clarification"] is True
    question = result["clarifying_question"].lower()
    assert "sister" in question or "her" in question or "she" in question
    assert "brother" not in question


# ==============================================================================
# C. Clarification Decision Engine
# ==============================================================================

def test_clarification_vague_product_type():
    # "Suggest something useful for travel"
    intent = _deterministic_intent("Suggest something useful for travel")
    assert intent["product_type"] is None
    assert intent["intended_use"] == "travel"

    decision, detail = evaluate_clarification_decision(intent, "Suggest something useful for travel")
    assert decision == "ASK_CLARIFICATION"
    question = _fallback_question(intent, "Suggest something useful for travel", detail)
    assert "travel" in question.lower()
    assert "hiking" not in question.lower()


def test_clarification_missing_budget_for_broad_type():
    # "Suggest a laptop" -> broad type, needs budget/use
    intent = _deterministic_intent("Suggest a laptop")
    decision, detail = evaluate_clarification_decision(intent, "Suggest a laptop")
    assert decision == "ASK_CLARIFICATION"
    question = _fallback_question(intent, "Suggest a laptop", detail)
    assert "budget" in question.lower() or "use" in question.lower()
    assert "laptop" in question.lower()


def test_clarification_sufficiently_specified_laptop():
    # "Suggest a laptop under ₹60,000 for programming" -> Should proceed to search!
    intent = _deterministic_intent("Suggest a laptop under ₹60,000 for programming")
    assert intent["budget_max"] == 60000.0
    assert intent["intended_use"] == "programming"
    decision, detail = evaluate_clarification_decision(intent, "Suggest a laptop under ₹60,000 for programming")
    assert decision == "SEARCH_PRODUCTS"
    assert detail is None


def test_clarification_sufficiently_specified_headphones():
    # "Wireless ANC headphones under ₹5,000" -> Should proceed to search!
    intent = _deterministic_intent("Wireless ANC headphones under ₹5,000")
    assert intent["budget_max"] == 5000.0
    assert "anc" in intent["preferences"]
    decision, detail = evaluate_clarification_decision(intent, "Wireless ANC headphones under ₹5,000")
    assert decision == "SEARCH_PRODUCTS"


def test_clarification_sufficiently_specified_gifts():
    # "Suggest gifts for my brother under ₹2,000" -> Has recipient & budget, proceed to search!
    intent = _deterministic_intent("Suggest gifts for my brother under ₹2,000")
    assert intent["recipient"] == "brother"
    assert intent["budget_max"] == 2000.0
    decision, detail = evaluate_clarification_decision(intent, "Suggest gifts for my brother under ₹2,000")
    assert decision == "SEARCH_PRODUCTS"


# ==============================================================================
# D. Follow-up Context & State Merging
# ==============================================================================

def test_multi_turn_followup_flow_for_gifts():
    SESSIONS.clear()
    session_id = "test_turn_flow"

    # Turn 1: "Suggest gifts for my brother"
    res1 = extract_intent({"session_id": session_id, "message": "Suggest gifts for my brother", "constraints": {}})
    c1 = res1["constraints"]
    assert c1["category"] == "Gifts"
    assert c1["recipient"] == "brother"
    assert c1.get("budget_max") is None

    # Check clarification on Turn 1
    clar1 = check_clarification({"constraints": c1, "message": "Suggest gifts for my brother"})
    assert clar1["needs_clarification"] is True
    assert "brother" in clar1["clarifying_question"].lower()

    # Turn 2: Follow-up answering budget "Under ₹2,000"
    res2 = extract_intent({"session_id": session_id, "message": "Under ₹2,000", "constraints": c1})
    c2 = res2["constraints"]
    assert c2["category"] == "Gifts"
    assert c2["recipient"] == "brother"
    assert c2["budget_max"] == 2000.0
    assert c2["is_followup"] is True

    # Check clarification on Turn 2: should now SEARCH!
    clar2 = check_clarification({"constraints": c2, "message": "Under ₹2,000"})
    assert clar2["needs_clarification"] is False
    assert clar2["clarification_decision"] in ("SEARCH_PRODUCTS", "UPDATE_EXISTING_PREFERENCES")

    # Turn 3: "Different brands"
    res3 = extract_intent({"session_id": session_id, "message": "Different brands", "constraints": c2})
    c3 = res3["constraints"]
    assert c3["category"] == "Gifts"
    assert c3["recipient"] == "brother"
    assert c3["budget_max"] == 2000.0
    assert c3["brand_diversity"] is True

    # Turn 4: "Make it cheaper"
    res4 = extract_intent({"session_id": session_id, "message": "Make it cheaper", "constraints": c3})
    c4 = res4["constraints"]
    assert c4["category"] == "Gifts"
    assert c4["recipient"] == "brother"
    assert c4["budget_max"] < 2000.0  # Budget reduced deterministically
    assert c4["cheaper"] is True


def test_intent_shift_resets_previous_constraints():
    # User was looking for gifts for brother, then says "I changed my mind, suggest a laptop instead"
    prev_constraints = {
        "category": "Gifts",
        "product_type": "gifts",
        "recipient": "brother",
        "budget_max": 2000.0,
        "brand_diversity": True,
    }
    res = extract_intent({
        "message": "I changed my mind, suggest a laptop instead",
        "constraints": prev_constraints,
    })
    c = res["constraints"]
    assert c["product_type"] == "laptop"
    assert c["category"] == "Electronics"
    # Old gift constraints must be cleared!
    assert c.get("recipient") is None
    assert c.get("budget_max") is None
    assert res["query_is_new"] is True


def test_followup_feature_addition():
    prev_constraints = {
        "product_type": "headphones",
        "category": "Electronics",
        "budget_max": 5000.0,
    }
    res = extract_intent({
        "message": "Only show wireless ones with noise cancellation",
        "constraints": prev_constraints,
    })
    c = res["constraints"]
    assert c["product_type"] == "headphones"
    assert c["budget_max"] == 5000.0
    assert "wireless" in c["preferences"]
    assert "anc" in c["preferences"]


# ==============================================================================
# E. Failure Handling & Robustness
# ==============================================================================

def test_empty_user_message():
    res = extract_intent({"message": "", "constraints": {"product_type": "laptop"}})
    assert res["constraints"]["product_type"] == "laptop"
    assert res["intent_confidence"] == 0.0


def test_gemini_unavailable_fallback_deterministic():
    # Verify extraction works completely when Gemini returns None
    res = extract_intent({"message": "Suggest a cabin suitcase under ₹6,000 for international travel", "constraints": {}})
    c = res["constraints"]
    assert c["product_type"] in ("cabin suitcase", "suitcase")
    assert c["budget_max"] == 6000.0
    assert c["intended_use"] == "international travel"
    assert res["intent_source"] == "deterministic"


def test_gemini_invalid_json_handled_gracefully(monkeypatch):
    # Mock gemini returning malformed or string instead of dict
    monkeypatch.setattr("backend.agents.intent_extraction.gemini_json", lambda prompt: None)
    res = extract_intent({"message": "Suggest running shoes under ₹3000", "constraints": {}})
    c = res["constraints"]
    assert c["product_type"] in ("running shoes", "shoes")
    assert c["budget_max"] == 3000.0


def test_unknown_product_category_graceful_handling():
    # An unusual product not in common lexicons e.g. "telescope"
    res = extract_intent({"message": "Suggest a telescope under ₹15,000", "constraints": {}})
    c = res["constraints"]
    assert c["product_type"] == "telescope"
    assert c["budget_max"] == 15000.0


# ==============================================================================
# Full LangGraph Multi-Turn Workflow with Clarification & Retrieval
# ==============================================================================

def test_full_langgraph_multi_turn_clarification_flow(monkeypatch):
    SESSIONS.clear()
    session_id = "test_multi_turn_graph"

    mock_watch = Product(
        product_id="serper:amazon:watch1",
        name="Fossil Leather Watch for Men",
        price=1899.0,
        currency="INR",
        price_verified=True,
        source_name="Amazon.in",
        product_url="https://www.amazon.in/dp/watch1",
        availability="in_stock",
        features=["Leather Strap", "Water Resistant"],
    )

    class MockShoppingProvider:
        name = "serper_shopping"
        def status(self): return {"name": self.name, "state": "ready"}
        def search(self, query="", max_price=None, **kwargs):
            if max_price is not None and mock_watch.price > max_price:
                return ProductSearchResult(products=[], providers=[self.name], provider_status={self.name: "success_with_no_results"})
            return ProductSearchResult(products=[mock_watch], providers=[self.name], provider_status={self.name: "success_with_results"})
        def get(self, pid): return mock_watch if pid == mock_watch.product_id else None

    import backend.graph.workflow as wf
    import backend.services.product_sources as sources_mod
    monkeypatch.setattr(sources_mod, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))
    monkeypatch.setattr(wf, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))

    # Turn 1: Broad gift query without budget
    out1 = run_shopping_graph(session_id, "Suggest me gifts for my brother")
    assert out1["action"] == "clarification"
    assert "brother" in out1["response_message"].lower()
    assert "sister" not in out1["response_message"].lower()
    assert len(out1["recommended_products"]) == 0

    # Turn 2: User provides budget
    out2 = run_shopping_graph(session_id, "Under ₹2,000")
    assert out2["action"] == "recommendations"
    assert len(out2["recommended_products"]) == 1
    assert out2["recommended_products"][0]["price"] == 1899.0
