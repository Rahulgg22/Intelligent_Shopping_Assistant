import json
from unittest.mock import MagicMock, patch
import pytest

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from backend.agents.clarification import check_clarification
from backend.agents.intent_extraction import extract_intent
from backend.agents.session_memory import SESSIONS
from backend.graph.workflow import run_shopping_graph
from backend.main import app
from backend.schemas.product import Product, ProductSearchResult
from backend.services.ai import (
    _clean_json_text,
    _get_configured_models,
    llm_state,
    openrouter_json,
)
from backend.services.product_retrieval import ProductRetrievalService


@pytest.fixture(autouse=True)
def mock_external_ml(monkeypatch):
    monkeypatch.setattr("backend.agents.ranking._cross_encoder_scores", lambda query, candidates: None)


# ==============================================================================
# Helper Mock Objects for OpenAI API responses
# ==============================================================================

def create_mock_completion(content: str):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# ==============================================================================
# 1. Model Configuration & JSON Cleaning
# ==============================================================================

def test_configured_models_ordering(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "deepseek/deepseek-chat,openai/gpt-4o-mini")
    models = _get_configured_models()
    assert models == ["google/gemini-2.5-flash", "deepseek/deepseek-chat", "openai/gpt-4o-mini"]


def test_structured_json_cleaning():
    # Clean standard markdown code fences
    raw1 = "```json\n{\"category\": \"Electronics\", \"product_type\": \"laptop\"}\n```"
    assert json.loads(_clean_json_text(raw1)) == {"category": "Electronics", "product_type": "laptop"}

    # Clean code fences without json tag
    raw2 = "```\n{\"budget_max\": 50000}\n```"
    assert json.loads(_clean_json_text(raw2)) == {"budget_max": 50000}

    # Extract JSON with conversational prose around it
    raw3 = "Here is the extracted shopping intent: {\"recipient\": \"brother\"} Hope that helps!"
    assert json.loads(_clean_json_text(raw3)) == {"recipient": "brother"}


# ==============================================================================
# 2. Successful OpenRouter Intent Extraction
# ==============================================================================

def test_successful_openrouter_intent_extraction(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mock-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")

    mock_llm_payload = {
        "category": "Fashion and Clothing",
        "subcategory": "Footwear",
        "product_type": "running shoes",
        "budget_max": 4000.0,
        "intended_use": "running",
        "preferences": ["lightweight"],
        "is_followup": False,
        "is_new_shopping_intent": True,
    }

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = create_mock_completion(json.dumps(mock_llm_payload))

        state = {
            "session_id": "test_openrouter_success",
            "message": "Suggest running shoes under ₹4,000",
            "normalized_query": "Suggest running shoes under ₹4,000",
            "constraints": {},
        }
        res = extract_intent(state)

        assert res["intent_source"] == "openrouter"
        assert res["intent_confidence"] == 0.95
        c = res["constraints"]
        assert c["product_type"] == "running shoes"
        assert c["budget_max"] == 4000.0
        assert c["intended_use"] == "running"


# ==============================================================================
# 3. Model Fallback Scenarios (Level A)
# ==============================================================================

def test_primary_model_failure_with_successful_fallback_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mock-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "deepseek/deepseek-chat")

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call (primary) raises HTTP 500 InternalServerError
        # Second call (fallback) succeeds
        mock_client.chat.completions.create.side_effect = [
            InternalServerError(
                message="Temporary internal error",
                response=MagicMock(status_code=500),
                body={"error": {"message": "Temporary internal error"}},
            ),
            create_mock_completion(json.dumps({"product_type": "backpack", "budget_max": 2500})),
        ]

        result = openrouter_json("Suggest a backpack under ₹2,500", operation="test_fallback")

        assert result == {"product_type": "backpack", "budget_max": 2500}
        assert mock_client.chat.completions.create.call_count == 2
        # Verify first call used primary model, second call used fallback model
        calls = mock_client.chat.completions.create.call_args_list
        assert calls[0].kwargs["model"] == "google/gemini-2.5-flash"
        assert calls[1].kwargs["model"] == "deepseek/deepseek-chat"


def test_http_429_and_503_transient_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mock-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "model-primary")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "model-fallback-1,model-fallback-2")

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Call 1: HTTP 429 RateLimitError
        # Call 2: HTTP 503 APIStatusError
        # Call 3: Success
        mock_client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}},
            ),
            APIStatusError(
                message="Service unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Service unavailable"}},
            ),
            create_mock_completion(json.dumps({"category": "Electronics"})),
        ]

        res = openrouter_json("Suggest gadgets", operation="test_429_503")
        assert res == {"category": "Electronics"}
        assert mock_client.chat.completions.create.call_count == 3


def test_invalid_api_key_aborts_immediately_without_retries(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-invalid-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "model-1")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "model-2,model-3")

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="Unauthorized: Invalid API Key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Unauthorized"}},
        )

        res = openrouter_json("Query", operation="test_auth_fail")

        # Must fail immediately and not make 3 useless calls
        assert res is None
        assert mock_client.chat.completions.create.call_count == 1


def test_malformed_json_moves_to_next_fallback_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mock-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "model-1")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "model-2")

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First model returns non-JSON text; second model returns valid JSON
        mock_client.chat.completions.create.side_effect = [
            create_mock_completion("Sorry, I cannot format this into JSON format."),
            create_mock_completion(json.dumps({"valid": True})),
        ]

        res = openrouter_json("Query", operation="test_malformed_json")
        assert res == {"valid": True}
        assert mock_client.chat.completions.create.call_count == 2


# ==============================================================================
# 4. Level B: Deterministic Fallback When All Models Fail
# ==============================================================================

def test_deterministic_fallback_when_all_models_fail(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mock-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "model-1")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "model-2")

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Both models fail with timeout
        mock_client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            APITimeoutError(request=MagicMock()),
        ]

        state = {
            "session_id": "test_all_fail",
            "message": "Find a laptop under ₹60,000 for programming",
            "normalized_query": "Find a laptop under ₹60,000 for programming",
            "constraints": {},
        }
        res = extract_intent(state)

        # Level B fallback must engage smoothly
        assert res["intent_source"] == "deterministic"
        c = res["constraints"]
        assert c["product_type"] == "laptop"
        assert c["budget_max"] == 60000.0
        assert c["intended_use"] == "programming"


def test_session_context_preservation_during_openrouter_failure(monkeypatch):
    # Turn 1 established gifts for brother
    prev_constraints = {
        "category": "Gifts",
        "product_type": "gifts",
        "recipient": "brother",
        "is_broad_category": False,
    }

    # Turn 2: OpenRouter is unconfigured or failing, user supplies "Under ₹2,000"
    with patch("backend.services.ai.openrouter_json", return_value=None):
        state = {
            "session_id": "test_ctx_preserve",
            "message": "Under ₹2,000",
            "normalized_query": "Under ₹2,000",
            "constraints": prev_constraints,
        }
        res = extract_intent(state)

        assert res["intent_source"] == "deterministic"
        c = res["constraints"]
        # Original context preserved and merged!
        assert c["recipient"] == "brother"
        assert c["category"] == "Gifts"
        assert c["budget_max"] == 2000.0
        assert c["is_followup"] is True


def test_followup_different_brands_during_fallback():
    prev_constraints = {
        "product_type": "laptop",
        "category": "Electronics",
        "budget_max": 50000.0,
    }

    with patch("backend.services.ai.openrouter_json", return_value=None):
        state = {
            "session_id": "test_diff_brands",
            "message": "Different brands",
            "normalized_query": "Different brands",
            "constraints": prev_constraints,
        }
        res = extract_intent(state)
        c = res["constraints"]
        assert c["product_type"] == "laptop"
        assert c["budget_max"] == 50000.0
        assert c["brand_diversity"] is True


def test_followup_make_it_cheaper_during_fallback():
    prev_constraints = {
        "product_type": "headphones",
        "category": "Electronics",
        "budget_max": 5000.0,
    }

    with patch("backend.services.ai.openrouter_json", return_value=None):
        state = {
            "session_id": "test_cheaper",
            "message": "Make it cheaper",
            "normalized_query": "Make it cheaper",
            "constraints": prev_constraints,
        }
        res = extract_intent(state)
        c = res["constraints"]
        assert c["product_type"] == "headphones"
        assert c["budget_max"] < 5000.0  # Reduced ~20%
        assert c["cheaper"] is True


# ==============================================================================
# 5. Clarification Behavior Under OpenRouter Integration
# ==============================================================================

def test_broad_request_clarification_under_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mock-test-key")

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = create_mock_completion(
            json.dumps({"question": "What kind of travel are you preparing for: holidays or backpacking?"})
        )

        state = {
            "session_id": "test_broad_clar",
            "message": "Suggest some travel equipment",
            "normalized_query": "Suggest some travel equipment",
            "constraints": {
                "product_type": "travel equipment",
                "category": "Travel and Luggage",
                "is_broad_category": True,
            },
        }
        clar_res = check_clarification(state)

        assert clar_res["needs_clarification"] is True
        assert "travel" in clar_res["clarifying_question"].lower()


# ==============================================================================
# 6. Full LangGraph Workflow with OpenRouter
# ==============================================================================

def test_full_langgraph_workflow_with_openrouter_mock(monkeypatch):
    SESSIONS.clear()
    session_id = "test_full_openrouter_flow"

    mock_laptop = Product(
        product_id="serper:amazon:dell_g15",
        name="Dell Gaming Laptop G15 Core i5",
        category="Electronics",
        subcategory="Laptops",
        price=54990.0,
        currency="INR",
        price_verified=True,
        source_name="Amazon.in",
        product_url="https://www.amazon.in/dp/dellg15",
        availability="in_stock",
        features=["16GB RAM", "512GB SSD", "RTX 3050"],
        use_cases=["gaming", "programming"],
    )

    class MockShoppingProvider:
        name = "serper_shopping"
        def status(self): return {"name": self.name, "state": "ready"}
        def search(self, query="", max_price=None, **kwargs):
            return ProductSearchResult(products=[mock_laptop], providers=[self.name], provider_status={self.name: "success_with_results"})
        def get(self, pid): return mock_laptop if pid == mock_laptop.product_id else None

    import backend.graph.workflow as wf
    import backend.services.product_sources as sources_mod
    monkeypatch.setattr(sources_mod, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))
    monkeypatch.setattr(wf, "build_product_service", lambda: ProductRetrievalService(MockShoppingProvider()))

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-mock-test-key")

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def mock_call(*args, **kwargs):
            messages = kwargs.get("messages", [])
            content = messages[0]["content"] if messages else ""
            if "shopping intent" in content.lower():
                return create_mock_completion(
                    json.dumps({
                        "category": "Electronics",
                        "subcategory": "Laptops",
                        "product_type": "laptop",
                        "budget_max": 60000.0,
                        "intended_use": "gaming",
                        "is_new_shopping_intent": True,
                    })
                )
            return create_mock_completion(
                json.dumps({
                    "explanations": [{"product_id": "serper:amazon:dell_g15", "explanation": "Top gaming laptop within budget."}]
                })
            )

        mock_client.chat.completions.create.side_effect = mock_call

        res = run_shopping_graph(session_id, "Find a gaming laptop under ₹60,000")

        assert res["action"] == "recommendations"
        assert res["intent_source"] == "openrouter"
        assert len(res["recommended_products"]) == 1
        assert res["recommended_products"][0]["name"] == "Dell Gaming Laptop G15 Core i5"
        assert res["recommended_products"][0]["price"] == 54990.0


# ==============================================================================
# 7. FastAPI /health Endpoint
# ==============================================================================

def test_health_endpoint_openrouter_status():
    from backend.main import health
    data = health()
    assert data["status"] == "healthy"
    assert "llm" in data
    assert data["llm"]["provider"] == "openrouter"
    assert "gemini" in data
    assert data["gemini"]["provider"] == "openrouter"
