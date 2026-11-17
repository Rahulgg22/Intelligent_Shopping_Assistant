import pytest
from unittest.mock import MagicMock

from backend.agents.intent_extraction import (
    extract_budget,
    extract_flexible_budget,
    extract_size,
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
def mock_external_ml_and_llm(monkeypatch):
    monkeypatch.setattr("backend.services.ai.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.intent_extraction.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.clarification.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.explanation.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.fallback.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.ranking._cross_encoder_scores", lambda query, candidates: None)


# ==============================================================================
# 1. Broad Shopping Categories (Must require clarification)
# ==============================================================================

def test_broad_travel_equipment_requires_clarification():
    intent = _deterministic_intent("Suggest some travel equipment")
    assert intent["product_type"] == "travel equipment"
    assert intent["category"] == "Travel and Luggage"
    assert intent["is_broad_category"] is True

    decision, detail = evaluate_clarification_decision(intent, "Suggest some travel equipment")
    assert decision == "ASK_CLARIFICATION"
    assert detail == "travel_equipment"

    q = _fallback_question(intent, "Suggest some travel equipment", detail)
    q_lower = q.lower()
    assert "travel" in q_lower
    assert any(term in q_lower for term in ("holidays", "backpacking", "business travel", "camping"))


def test_broad_electronics_requires_clarification():
    intent = _deterministic_intent("Suggest some electronics")
    assert intent["product_type"] == "electronics"
    assert intent["category"] == "Electronics"
    assert intent["is_broad_category"] is True

    decision, detail = evaluate_clarification_decision(intent, "Suggest some electronics")
    assert decision == "ASK_CLARIFICATION"
    assert detail == "electronics"

    q = _fallback_question(intent, "Suggest some electronics", detail)
    assert "electronic" in q.lower() or "laptop" in q.lower() or "smartphone" in q.lower()


def test_broad_furniture_requires_clarification():
    intent = _deterministic_intent("I need furniture")
    decision, detail = evaluate_clarification_decision(intent, "I need furniture")
    assert decision == "ASK_CLARIFICATION"
    q = _fallback_question(intent, "I need furniture", detail)
    assert "furniture" in q.lower()


def test_broad_skincare_requires_clarification():
    intent = _deterministic_intent("Recommend skincare products")
    decision, detail = evaluate_clarification_decision(intent, "Recommend skincare products")
    assert decision == "ASK_CLARIFICATION"
    q = _fallback_question(intent, "Recommend skincare products", detail)
    assert "skincare" in q.lower()


def test_broad_camping_gear_requires_clarification():
    intent = _deterministic_intent("Suggest some camping gear")
    decision, detail = evaluate_clarification_decision(intent, "Suggest some camping gear")
    assert decision == "ASK_CLARIFICATION"


# ==============================================================================
# 2. Specific Products Without Budget or Key Details
# ==============================================================================

def test_bare_backpack_requires_clarification():
    intent = _deterministic_intent("Suggest a backpack")
    assert intent["product_type"] == "backpack"
    assert intent["budget_max"] is None
    assert intent["intended_use"] is None

    decision, detail = evaluate_clarification_decision(intent, "Suggest a backpack")
    assert decision == "ASK_CLARIFICATION"
    assert detail == "backpack"

    q = _fallback_question(intent, "Suggest a backpack", detail)
    q_lower = q.lower()
    assert "backpack" in q_lower
    assert "budget" in q_lower or "college" in q_lower or "travel" in q_lower


def test_suitcase_for_international_travel_requires_clarification():
    # Suitcases vary dramatically by size (cabin vs check-in) and price (₹2,000 - ₹30,000)
    intent = _deterministic_intent("Suggest a suitcase for international travel")
    assert intent["product_type"] == "suitcase"
    assert intent["intended_use"] == "international travel"
    assert intent["budget_max"] is None
    assert intent["size"] is None

    decision, detail = evaluate_clarification_decision(intent, "Suggest a suitcase for international travel")
    assert decision == "ASK_CLARIFICATION"
    assert detail == "suitcase"

    q = _fallback_question(intent, "Suggest a suitcase for international travel", detail)
    q_lower = q.lower()
    assert "budget" in q_lower or "size" in q_lower or "cabin" in q_lower


def test_bare_headphones_requires_clarification():
    intent = _deterministic_intent("I need headphones")
    decision, detail = evaluate_clarification_decision(intent, "I need headphones")
    assert decision == "ASK_CLARIFICATION"
    q = _fallback_question(intent, "I need headphones", detail)
    assert "headphones" in q.lower()
    assert "budget" in q.lower() or "use" in q.lower()


def test_gifts_for_brother_without_budget_requires_clarification():
    intent = _deterministic_intent("Suggest gifts for my brother")
    assert intent["recipient"] == "brother"
    assert intent["budget_max"] is None

    decision, detail = evaluate_clarification_decision(intent, "Suggest gifts for my brother")
    assert decision == "ASK_CLARIFICATION"
    q = _fallback_question(intent, "Suggest gifts for my brother", detail)
    q_lower = q.lower()
    assert "brother" in q_lower
    assert "sister" not in q_lower
    assert "interests" in q_lower or "budget" in q_lower


# ==============================================================================
# 3. Sufficiently Specific Queries (Proceed Directly to Retrieval)
# ==============================================================================

def test_trekking_backpack_with_size_and_budget_proceeds_to_search():
    intent = _deterministic_intent("Suggest a 40L backpack for trekking under ₹3,000")
    assert intent["product_type"] == "backpack"
    assert intent["size"] == "40l"
    assert intent["intended_use"] in ("trekking", "hiking")
    assert intent["budget_max"] == 3000.0

    decision, detail = evaluate_clarification_decision(intent, "Suggest a 40L backpack for trekking under ₹3,000")
    assert decision == "SEARCH_PRODUCTS"
    assert detail is None


def test_wireless_anc_headphones_with_budget_proceeds_to_search():
    intent = _deterministic_intent("Suggest wireless headphones under ₹5,000 with ANC")
    assert intent["budget_max"] == 5000.0
    assert "anc" in intent["preferences"]
    assert "wireless" in intent["preferences"]

    decision, detail = evaluate_clarification_decision(intent, "Suggest wireless headphones under ₹5,000 with ANC")
    assert decision == "SEARCH_PRODUCTS"
    assert detail is None


def test_cabin_suitcase_with_budget_proceeds_to_search():
    intent = _deterministic_intent("Suggest a cabin suitcase under ₹6,000 for international travel")
    assert intent["product_type"] == "cabin suitcase"
    assert intent["budget_max"] == 6000.0
    assert intent["intended_use"] == "international travel"

    decision, detail = evaluate_clarification_decision(intent, "Suggest a cabin suitcase under ₹6,000 for international travel")
    assert decision == "SEARCH_PRODUCTS"
    assert detail is None


# ==============================================================================
# 4. Explicit Flexible Budget Instructions ('any budget', 'surprise me')
# ==============================================================================

@pytest.mark.parametrize("query", [
    "Suggest a suitcase for international travel, any budget",
    "Suggest a suitcase for international travel, no budget limit",
    "Suggest a backpack for college, surprise me",
    "Suggest running shoes, show me anything",
    "Suggest a laptop for programming, budget doesn't matter",
])
def test_explicit_flexible_budget_bypasses_budget_clarification(query):
    intent = _deterministic_intent(query)
    assert intent["flexible_budget"] is True
    assert intent["budget_max"] is None

    decision, detail = evaluate_clarification_decision(intent, query)
    assert decision in ("SEARCH_PRODUCTS", "UPDATE_EXISTING_PREFERENCES")


# ==============================================================================
# 5. Workflow Verification: Retrieval Must NOT be Called on Clarification
# ==============================================================================

def test_clarification_prevents_premature_provider_retrieval(monkeypatch):
    SESSIONS.clear()
    session_id = "test_no_premature_retrieval"

    search_spy = MagicMock()

    class MockShoppingProvider:
        name = "serper_shopping"
        def status(self): return {"name": self.name, "state": "ready"}
        def search(self, *args, **kwargs):
            search_spy(*args, **kwargs)
            return ProductSearchResult(products=[], providers=[self.name])
        def get(self, pid): return None

    import backend.graph.workflow as wf
    import backend.services.product_sources as sources_mod
    monkeypatch.setattr(sources_mod, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))
    monkeypatch.setattr(wf, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))

    # Broad query: "Suggest some travel equipment"
    result = run_shopping_graph(session_id, "Suggest some travel equipment")

    # 1. Action must be "clarification"
    assert result["action"] == "clarification"
    assert len(result.get("recommended_products", [])) == 0
    assert "travel" in result["response_message"].lower()

    # 2. External retrieval MUST NOT have been called!
    assert search_spy.call_count == 0, "External search was prematurely triggered during a clarification turn!"


# ==============================================================================
# 6. Workflow Verification: Multi-Turn Turn 1 -> Turn 2 Flow
# ==============================================================================

def test_full_multiturn_flow_travel_equipment_to_specific_product(monkeypatch):
    SESSIONS.clear()
    session_id = "test_multiturn_travel"

    mock_backpack = Product(
        product_id="serper:amazon:bp1",
        name="Wildcraft 40L Trekking Backpack",
        price=2499.0,
        currency="INR",
        price_verified=True,
        source_name="Amazon.in",
        product_url="https://www.amazon.in/dp/bp1",
        availability="in_stock",
        features=["40L Capacity", "Water Resistant", "Rain Cover"],
        use_cases=["trekking", "travel"],
    )

    search_spy = MagicMock()

    class MockShoppingProvider:
        name = "serper_shopping"
        def status(self): return {"name": self.name, "state": "ready"}
        def search(self, query="", max_price=None, **kwargs):
            search_spy(query=query, max_price=max_price)
            if max_price is not None and mock_backpack.price > max_price:
                return ProductSearchResult(products=[], providers=[self.name], provider_status={self.name: "success_with_no_results"})
            return ProductSearchResult(products=[mock_backpack], providers=[self.name], provider_status={self.name: "success_with_results"})
        def get(self, pid): return mock_backpack if pid == mock_backpack.product_id else None

    import backend.graph.workflow as wf
    import backend.services.product_sources as sources_mod
    monkeypatch.setattr(sources_mod, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))
    monkeypatch.setattr(wf, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))

    # Turn 1: Broad query
    turn1 = run_shopping_graph(session_id, "Suggest some travel equipment")
    assert turn1["action"] == "clarification"
    assert search_spy.call_count == 0
    assert len(turn1.get("recommended_products", [])) == 0

    # Turn 2: User provides specific item, capacity, use, and budget
    turn2 = run_shopping_graph(session_id, "I need a 40L backpack for trekking under ₹3,000")
    assert turn2["action"] == "recommendations"
    assert search_spy.call_count == 1
    assert len(turn2["recommended_products"]) == 1
    assert turn2["recommended_products"][0]["name"] == "Wildcraft 40L Trekking Backpack"
    assert turn2["recommended_products"][0]["price"] == 2499.0


def test_multiturn_suitcase_flow(monkeypatch):
    SESSIONS.clear()
    session_id = "test_multiturn_suitcase"

    mock_suitcase = Product(
        product_id="serper:flipkart:sc1",
        name="American Tourister 55cm Cabin Trolley Suitcase",
        price=4299.0,
        currency="INR",
        price_verified=True,
        source_name="Flipkart",
        product_url="https://www.flipkart.com/dp/sc1",
        availability="in_stock",
        features=["Cabin Luggage", "4 Wheels", "TSA Lock"],
        use_cases=["international travel"],
    )

    class MockShoppingProvider:
        name = "serper_shopping"
        def status(self): return {"name": self.name, "state": "ready"}
        def search(self, query="", max_price=None, **kwargs):
            if max_price is not None and mock_suitcase.price > max_price:
                return ProductSearchResult(products=[], providers=[self.name], provider_status={self.name: "success_with_no_results"})
            return ProductSearchResult(products=[mock_suitcase], providers=[self.name], provider_status={self.name: "success_with_results"})
        def get(self, pid): return mock_suitcase if pid == mock_suitcase.product_id else None

    import backend.graph.workflow as wf
    import backend.services.product_sources as sources_mod
    monkeypatch.setattr(sources_mod, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))
    monkeypatch.setattr(wf, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))

    # Turn 1: "Suggest a suitcase for international travel" -> needs budget or size
    turn1 = run_shopping_graph(session_id, "Suggest a suitcase for international travel")
    assert turn1["action"] == "clarification"
    assert len(turn1.get("recommended_products", [])) == 0

    # Turn 2: User provides size and budget
    turn2 = run_shopping_graph(session_id, "Cabin size under ₹6,000")
    assert turn2["action"] == "recommendations"
    assert len(turn2["recommended_products"]) == 1
    assert turn2["recommended_products"][0]["price"] == 4299.0


def test_avoid_repeating_same_clarification_question():
    constraints = {
        "product_type": "backpack",
        "last_clarifying_question": "What will you primarily use the backpack for (such as college, travel, or trekking), and do you have an approximate budget?",
    }
    state = {
        "constraints": constraints,
        "message": "Just show me the best ones",
        "pending_clarification_question": constraints["last_clarifying_question"],
    }
    res = check_clarification(state)
    assert res["needs_clarification"] is False
    assert res["clarification_decision"] == "SEARCH_PRODUCTS"
