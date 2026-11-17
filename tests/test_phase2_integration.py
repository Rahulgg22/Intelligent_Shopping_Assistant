import pytest
from backend.agents.session_memory import SESSIONS
from backend.graph.workflow import provider_retrieval_node, run_shopping_graph
from backend.mcp_server import get_product_details, get_provider_status, search_products
from backend.providers.composite import CompositeProductProvider


@pytest.fixture(autouse=True)
def mock_external_ml_and_llm(monkeypatch):
    monkeypatch.setattr("backend.services.ai.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.intent_extraction.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.clarification.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.explanation.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.fallback.gemini_json", lambda prompt: None)
    monkeypatch.setattr("backend.agents.ranking._cross_encoder_scores", lambda query, candidates: None)


def test_graph_retrieval_node_reports_external_provider_status():
    result = provider_retrieval_node({
        "normalized_query": "laptop under ₹60000 for programming",
        "mapped_filters": {
            "category": "laptops and electronics",
            "subcategory": "Laptops",
            "max_price": 60000,
            "in_stock_only": True,
        },
    })
    assert "tavily_web_search" in result["provider_status"]


def test_mcp_tools_report_external_search_status():
    result = search_products("programming laptop", category="Electronics", subcategory="Laptops", max_price=60000)
    assert "products" in result
    assert get_provider_status()["providers"]["tavily_web_search"]["state"] in {"ready", "missing_credentials"}
    assert get_product_details("unknown-product")["product"] is None


def test_mcp_budget_and_empty_results_are_explicit():
    result = search_products("laptop", category="Electronics", subcategory="Laptops", max_price=1)
    assert result["products"] == []
    assert result["errors"] or result["provider_status"]


def test_full_graph_uses_provider_service_for_natural_language_query():
    SESSIONS.clear()
    result = run_shopping_graph("phase2", "Find a laptop under ₹60000 for programming")
    assert "tavily_web_search" in result["provider_status"]
    assert result["action"] in {"message", "recommendations"}