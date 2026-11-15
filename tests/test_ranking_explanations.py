import pytest

from backend.agents import ranking
from backend.agents.explanation import explain_recommendations
from backend.api.routes.chat import chat
from backend.schemas.chat import ChatRequest
from backend.schemas.product import Product, ProductSearchResult


def product(identifier, **changes):
    return {"product_id": identifier, "name": "Cabin suitcase", "category": "travel and luggage",
            "description": "", "price": 3000, "price_verified": True, "availability": "unknown",
            "availability_verified": False, **changes}


def run(monkeypatch, products, constraints=None, scores=None):
    monkeypatch.setattr(ranking, "_cross_encoder_scores", lambda query, items: scores)
    state = {"candidates": products, "constraints": constraints or {"budget_max": 6000},
             "message": "cabin suitcase", "normalized_query": "cabin suitcase"}
    state.update(ranking.rank_candidates(state))
    state.update(explain_recommendations(state))
    return state


def test_weighted_order_and_explanation_advantage(monkeypatch):
    state = run(monkeypatch, [product("plain"), product("lock", name="Cabin suitcase TSA lock")],
                {"budget_max": 6000, "preferences": ["tsa lock"]}, [0, 0])
    first, second = state["recommended_products"]
    assert first["product_id"] == "lock"
    assert state["ranking_weights"] == {"relevance": .7, "rating": .1, "preference": .2}
    assert first["rank_score"] == pytest.approx(sum(s["contribution"] for s in first["rank_signals"].values()))
    assert first["rank_signals"]["preference"]["contribution"] > second["rank_signals"]["preference"]["contribution"]
    assert "preferences" in first["rank_explanation"]["comparison"]
    assert "tsa lock" in first["explanation"]


def test_budget_verification_and_stock(monkeypatch):
    state = run(monkeypatch, [product("ok"), product("cheap", price=900), product("dear", price=7000),
                              product("unknown", price=None), product("unverified", price_verified=False)],
                {"budget_min": "1000", "budget_max": "6000"})
    assert [p["product_id"] for p in state["ranked_products"]] == ["ok"]
    assert state["filter_stats"]["unverified_price"] == 2
    assert state["filter_stats"]["under_min"] == 1
    assert state["filter_stats"]["over_budget"] == 1
    assert "only eligible" in state["recommended_products"][0]["explanation"]
    assert state["recommended_products"][0]["rank_explanation"]["comparison"] == ""
    assert "Stock has not been confirmed" in state["recommended_products"][0]["tradeoffs"][1]


def test_tie_neutral_rating_and_active_weights(monkeypatch):
    state = run(monkeypatch, [product("a", rating=5, review_count=1), product("b")], scores=[0, 0])
    assert state["ranking_weights"]["relevance"] == pytest.approx(.875)
    assert "preference" not in state["ranking_weights"]
    assert [p["product_id"] for p in state["ranked_products"]] == ["a", "b"]
    assert "Tied" in state["recommended_products"][0]["rank_explanation"]["comparison"]
    assert state["ranked_products"][0]["rank_signals"]["rating"]["neutral"]


def test_shortlist_and_uniform_fallback(monkeypatch):
    state = run(monkeypatch, [product(str(i)) for i in range(40)], scores=[float('nan')] * 30)
    assert len(state["ranked_products"]) == 30
    assert state["filter_stats"]["outside_shortlist"] == 10
    assert state["ranking_mode"] == "lexical_fallback"
    assert len({p["rank_score"] for p in state["ranked_products"]}) == 1
    assert "text matching" in state["ranking_explanation"]["summary"]


def test_stock_constraint_and_minimum_only(monkeypatch):
    state = run(monkeypatch, [product("unknown"), product("stock", availability="in_stock", availability_verified=True)],
                {"budget_min": 1000, "hard_constraints": ["in_stock"]})
    assert len(state["ranked_products"]) == 1
    assert state["filter_stats"]["unconfirmed_stock"] == 1
    assert "at least" in state["recommended_products"][0]["rank_explanation"]["budget"]


def test_fastapi_full_graph_explanations(monkeypatch):
    monkeypatch.setattr("backend.agents.intent_extraction.gemini_json", lambda prompt: None)
    monkeypatch.setattr(ranking, "_cross_encoder_scores", lambda query, items: [0] * len(items))
    class Service:
        def search(self, *args, **kwargs):
            return ProductSearchResult(products=[Product(**product("lock", name="Cabin suitcase TSA lock")),
                                                  Product(**product("plain"))], provider_status={"mock": "ready"})
    monkeypatch.setattr("backend.graph.workflow.build_product_service", Service)
    response = chat(ChatRequest(session_id="explanation-milestone",
                                message="a cabin suitcase under \u20b96,000 with a TSA lock"))
    data = response.model_dump(mode="json")
    assert data["products"][0]["product_id"] == "lock"
    assert data["products"][0]["rank"] == 1
    assert data["products"][0]["in_stock"] is None
    assert data["ranking_explanation"]["source"] == "deterministic"
    assert data["products"][0]["rank_explanation"]["signals"]["preference"]["matched_terms"] == ["tsa lock"]
