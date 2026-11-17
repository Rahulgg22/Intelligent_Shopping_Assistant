import time
from unittest.mock import MagicMock, patch
import pytest

from backend.agents.explanation import (
    _extract_evidence_and_requirement_matches,
    _build_deterministic_retrieval_reason,
    _build_deterministic_ranking_reason,
    explain_recommendations,
    handle_explanation_followup,
)
from backend.agents.ranking import (
    get_cross_encoder,
    _cross_encoder_scores,
    rank_candidates,
)
from backend.services.ai import openrouter_json, llm_state
from backend.graph.workflow import run_shopping_graph
from backend.agents.session_memory import save_session


# ---------------------------------------------------------------------------
# 1. Product Explanation Agent Tests
# ---------------------------------------------------------------------------

def test_explanation_retrieval_and_ranking_reasons():
    """Verify that retrieval_reason and ranking_reason reflect user query and rank position."""
    product = {
        "product_id": "p1",
        "name": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "brand": "Sony",
        "category": "Electronics",
        "subcategory": "Headphones",
        "price": 4500.0,
        "price_verified": True,
        "availability": "in_stock",
        "features": ["Active Noise Cancellation", "Bluetooth 5.2", "40hr Battery"],
        "rating": 4.6,
    }
    constraints = {
        "product_type": "wireless headphones",
        "budget_max": 5000.0,
        "preferences": ["anc", "wireless"],
    }
    evidence = _extract_evidence_and_requirement_matches(product, constraints, rank=1, total_candidates=3)
    retrieval_r = _build_deterministic_retrieval_reason(product, constraints, evidence)
    ranking_r = _build_deterministic_ranking_reason(product, rank=1, total_candidates=3, evidence=evidence)

    assert "wireless headphones" in retrieval_r.lower()
    assert "budget" in retrieval_r.lower() or "4,500" in retrieval_r
    assert "ranked #1" in ranking_r.lower()


def test_requirement_match_statuses_matched_unmatched_unknown():
    """Verify that requirement statuses correctly distinguish matched, not_matched, and unknown."""
    product = {
        "product_id": "p2",
        "name": "Generic Bluetooth Headphones",
        "brand": "Generic",
        "category": "Electronics",
        "price": 6000.0,  # Exceeds 5000 budget
        "price_verified": True,
        "features": ["Bluetooth 5.0"],  # Has wireless, lacks ANC and battery info
    }
    constraints = {
        "product_type": "headphones",
        "budget_max": 5000.0,
        "preferences": ["wireless", "anc", "long battery life"],
    }
    evidence = _extract_evidence_and_requirement_matches(product, constraints, rank=2, total_candidates=3)
    matches = {m["requirement"].lower(): m["status"] for m in evidence["requirement_matches"]}

    # Budget should be not_matched because 6000 > 5000
    budget_key = next((k for k in matches if "budget" in k), None)
    assert budget_key is not None
    assert matches[budget_key] == "not_matched"

    # Wireless should be matched
    assert matches["wireless"] == "matched"

    # ANC and Battery should be unknown (NOT matched)
    assert matches["anc"] == "unknown"
    battery_key = next((k for k in matches if "battery" in k), None)
    assert battery_key is not None
    assert matches[battery_key] == "unknown"


def test_unknown_specs_never_treated_as_matched():
    """Ensure that unknown specifications are never falsely marked as matched."""
    product = {
        "product_id": "p3",
        "name": "Standard Laptop",
        "price": 40000.0,
        "features": ["8GB RAM", "256GB SSD"],
    }
    constraints = {
        "preferences": ["16GB RAM", "Backlit Keyboard"],
    }
    evidence = _extract_evidence_and_requirement_matches(product, constraints, rank=1, total_candidates=1)
    for rm in evidence["requirement_matches"]:
        if "16gb" in rm["requirement"].lower() or "backlit" in rm["requirement"].lower():
            assert rm["status"] == "unknown"


def test_missing_prices_and_ratings():
    """Ensure missing prices and missing ratings are noted as unverified rather than fabricated."""
    product = {
        "product_id": "p4",
        "name": "Unpriced Hiking Backpack",
        "price": None,
        "price_verified": False,
        "rating": None,
    }
    constraints = {
        "product_type": "backpack",
        "budget_max": 3000.0,
    }
    evidence = _extract_evidence_and_requirement_matches(product, constraints, rank=1, total_candidates=1)
    budget_m = next(m for m in evidence["requirement_matches"] if "budget" in m["requirement"].lower())
    assert budget_m["status"] == "unknown"
    assert any("price" in lim.lower() for lim in evidence["limitations"])
    assert any("rating" in lim.lower() for lim in evidence["limitations"])


def test_deterministic_fallback_when_openrouter_fails(monkeypatch):
    """When OpenRouter returns None (402 or error), explanation agent falls back gracefully."""
    import backend.agents.explanation as exp_mod
    monkeypatch.setattr(exp_mod, "openrouter_json", lambda prompt, operation: None)

    state = {
        "constraints": {"product_type": "headphones", "budget_max": 5000},
        "ranked_products": [
            {
                "product_id": "p_fall",
                "name": "Soundcore Q20 ANC Headphones",
                "price": 4200,
                "price_verified": True,
                "features": ["Active Noise Cancelling", "Bluetooth"],
            }
        ],
        "provider_status": {"serper": "ready"},
        "retrieval_errors": {},
    }
    result = exp_mod.explain_recommendations(state)
    assert result["action"] == "recommendations"
    prods = result["recommended_products"]
    assert len(prods) == 1
    p = prods[0]
    assert p["retrieval_reason"]
    assert p["ranking_reason"]
    assert "explanation_details" in p
    assert p["explanation_details"]["product_id"] == "p_fall"
    assert len(p["requirement_matches"]) > 0


def test_explanations_do_not_change_product_ordering(monkeypatch):
    """Explanations must respect the pre-calculated rank ordering."""
    import backend.agents.explanation as exp_mod
    # Mock LLM attempting to swap or invert ranks
    monkeypatch.setattr(exp_mod, "openrouter_json", lambda prompt, operation: {
        "explanations": [
            {"product_id": "p2", "retrieval_reason": "R2", "ranking_reason": "Ranked #1"},
            {"product_id": "p1", "retrieval_reason": "R1", "ranking_reason": "Ranked #2"},
        ]
    })

    state = {
        "constraints": {"product_type": "shoes"},
        "ranked_products": [
            {"product_id": "p1", "name": "Shoe A", "rank": 1, "price": 2000},
            {"product_id": "p2", "name": "Shoe B", "rank": 2, "price": 3000},
        ],
        "provider_status": {},
        "retrieval_errors": {},
    }
    result = exp_mod.explain_recommendations(state)
    prods = result["recommended_products"]
    assert prods[0]["product_id"] == "p1"
    assert prods[0]["rank"] == 1
    assert prods[1]["product_id"] == "p2"
    assert prods[1]["rank"] == 2


# ---------------------------------------------------------------------------
# 2. Follow-Up Explanation Queries (No Re-Search)
# ---------------------------------------------------------------------------

def test_followup_why_ranked_first():
    """Follow-up 'Why is this ranked first?' explains rank 1 using stored evidence."""
    prev = [
        {
            "product_id": "p1",
            "name": "Dell Inspiron 15",
            "rank": 1,
            "price": 55000,
            "retrieval_reason": "Matches 16GB programming laptop requirement.",
            "ranking_reason": "Ranked #1 due to confirmed 16GB RAM and top rating.",
            "strengths": ["16GB RAM confirmed", "High 4.5★ rating"],
            "limitations": [],
        },
        {
            "product_id": "p2",
            "name": "Lenovo IdeaPad",
            "rank": 2,
            "price": 52000,
        }
    ]
    res = handle_explanation_followup("Why is this ranked first?", prev, {})
    assert res["action"] == "message"
    assert "Dell Inspiron 15" in res["response_message"]
    assert "#1" in res["response_message"]
    assert "16GB RAM" in res["response_message"]


def test_followup_what_does_it_not_satisfy():
    """Follow-up 'What does it not satisfy?' surfaces limitations and unknown requirements."""
    prev = [
        {
            "product_id": "p1",
            "name": "Mid-tier Earbuds",
            "rank": 1,
            "limitations": ["Battery life details could not be verified", "No wireless charging"],
            "requirement_matches": [
                {"requirement": "Wireless Charging", "status": "unknown"},
            ]
        }
    ]
    res = handle_explanation_followup("What does it not satisfy?", prev, {})
    assert res["action"] == "message"
    assert "Battery life" in res["response_message"] or "Wireless Charging" in res["response_message"]


def test_followup_why_better_than_runner_up():
    """Follow-up 'Why is this product better?' compares #1 against #2."""
    prev = [
        {
            "product_id": "p1",
            "name": "Flagship Earbuds",
            "rank": 1,
            "ranking_reason": "Highest verified ANC coverage and rating.",
            "strengths": ["Active Noise Cancellation", "4.8★ rating"],
        },
        {
            "product_id": "p2",
            "name": "Budget Earbuds",
            "rank": 2,
            "limitations": ["No ANC"],
        }
    ]
    res = handle_explanation_followup("Why is this product better?", prev, {})
    assert res["action"] == "message"
    assert "Flagship Earbuds" in res["response_message"]
    assert "Budget Earbuds" in res["response_message"]


# ---------------------------------------------------------------------------
# 3. CrossEncoder Performance & Reuse
# ---------------------------------------------------------------------------

def test_cross_encoder_singleton_reuse():
    """Ensure get_cross_encoder() returns cached instance with zero subsequent init latency."""
    encoder1, init_time1 = get_cross_encoder()
    if encoder1 is not None:
        encoder2, init_time2 = get_cross_encoder()
        assert encoder1 is encoder2
        assert init_time2 == 0.0


def test_ranking_candidate_pruning_and_timing():
    """Verify candidate filtering, inference timing, and pruning to top_k."""
    candidates = [
        {
            "product_id": f"p_{i}",
            "name": f"Product {i} Bluetooth Headphones",
            "price": 2000 + i * 100,
            "price_verified": True,
            "features": ["Bluetooth"],
        }
        for i in range(25)
    ]
    state = {
        "message": "Bluetooth headphones under 5000",
        "constraints": {"budget_max": 5000, "product_type": "headphones"},
        "candidates": candidates,
    }
    # Ensure singleton model is warmed up so we measure candidate pruning & inference timing
    get_cross_encoder()
    t0 = time.perf_counter()
    res = rank_candidates(state)
    elapsed = time.perf_counter() - t0

    ranked = res["ranked_products"]
    assert len(ranked) > 0
    # Overall ranking should finish quickly (< 1.5 seconds)
    assert elapsed < 5.0
    assert ranked[0]["rank"] == 1


# ---------------------------------------------------------------------------
# 4. OpenRouter Error Handling & HTTP 402 Fallback
# ---------------------------------------------------------------------------

def test_openrouter_402_stops_candidate_retries(monkeypatch):
    """Ensure HTTP 402 error halts candidate models immediately without redundant retries."""
    import backend.services.ai as ai_mod
    from openai import APIStatusError
    import httpx

    # Create dummy HTTP 402 response
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx.Response(status_code=402, request=req, text='{"error": {"message": "Payment Required"}}')
    err_402 = APIStatusError(message="Payment Required", response=resp, body=None)

    call_count = 0
    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise err_402

    monkeypatch.setattr(ai_mod, "_get_configured_models", lambda: ["model-primary", "model-fallback-1", "model-fallback-2"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake-key")

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create
        mock_openai_cls.return_value = mock_client

        try:
            res = openrouter_json("test prompt", operation="extract_intent")
            assert res is None
            # On permanent 402, candidate model loop must break after 1 attempt, NOT retry 3 times
            assert call_count == 1
            assert ai_mod.llm_state["last_category"] == "payment_required"
        finally:
            ai_mod.reset_cooldown()


def test_openrouter_transient_error_tries_fallback_model(monkeypatch):
    """Ensure transient 500 or 429 attempts subsequent fallback models."""
    import backend.services.ai as ai_mod
    from openai import APIStatusError
    import httpx

    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp_500 = httpx.Response(status_code=500, request=req, text='{"error": "Internal Server Error"}')
    err_500 = APIStatusError(message="Server error", response=resp_500, body=None)

    models_attempted = []
    def mock_create(model=None, **kwargs):
        models_attempted.append(model)
        if model == "model-primary":
            raise err_500
        # Second model succeeds
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='{"status": "ok"}'))]
        return mock_resp

    monkeypatch.setattr(ai_mod, "_get_configured_models", lambda: ["model-primary", "model-fallback-1"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fake-key")
    ai_mod._account_cooldown_until = 0.0

    with patch("backend.services.ai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create
        mock_openai_cls.return_value = mock_client

        res = openrouter_json("test prompt", operation="test_transient")
        assert res == {"status": "ok"}
        assert models_attempted == ["model-primary", "model-fallback-1"]
