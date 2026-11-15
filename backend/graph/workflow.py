import logging
import time

from langgraph.graph import END, START, StateGraph

from backend.agents.attribute_mapping import map_attributes
from backend.agents.clarification import check_clarification
from backend.agents.explanation import explain_recommendations
from backend.agents.fallback import handle_fallback
from backend.agents.intent_extraction import extract_intent
from backend.agents.query_ingestion import ingest
from backend.agents.ranking import rank_candidates
from backend.services.product_sources import build_product_service
from backend.agents.session_memory import load_session, save_session
from backend.graph.state import ShoppingState

timing_logger = logging.getLogger("timing")


def timed(name, fn):
    def wrapped(state):
        started = time.perf_counter()
        try:
            return fn(state)
        finally:
            timing_logger.info("node=%s took=%.2fs", name, time.perf_counter() - started)
    return wrapped

build_catalog_service = build_product_service


def route_after_clarification(state):
    return "clarification" if state.get("needs_clarification") else "mapping"


def route_after_ranking(state):
    return "explanation" if state.get("ranked_products") else "fallback"


def clarification_node(state):
    return {
        "action": "clarification",
        "response_message": state.get("clarifying_question"),
        "recommended_products": [],
    }


def explanation_node(state):
    return explain_recommendations(state)


def provider_retrieval_node(state):
    filters = state.get("mapped_filters", {})
    result = build_product_service().search(
        state.get("normalized_query", state.get("message", "")),
        category=filters.get("category"),
        subcategory=filters.get("subcategory"),
        min_price=state.get("constraints", {}).get("budget_min"),
        max_price=filters.get("max_price"),
        in_stock_only="in_stock" in state.get("constraints", {}).get("hard_constraints", []),
        limit=50,
    )
    return {
        "candidates": [product.model_dump(mode="json") for product in result.products],
        "provider_status": result.provider_status,
        "retrieval_errors": result.errors,
    }


def build_graph():
    graph = StateGraph(ShoppingState)
    nodes = {
        "load_session": load_session, "ingest": ingest, "extract_intent": extract_intent,
        "check_clarification": check_clarification, "clarification": clarification_node,
        "mapping": map_attributes, "retrieval": provider_retrieval_node, "ranking": rank_candidates,
        "explanation": explanation_node, "fallback": handle_fallback, "save_session": save_session,
    }
    for name, fn in nodes.items():
        graph.add_node(name, timed(name, fn))

    graph.add_edge(START, "load_session")
    graph.add_edge("load_session", "ingest")
    graph.add_edge("ingest", "extract_intent")
    graph.add_edge("extract_intent", "check_clarification")
    graph.add_conditional_edges("check_clarification", route_after_clarification)
    graph.add_edge("clarification", "save_session")
    graph.add_edge("mapping", "retrieval")
    graph.add_edge("retrieval", "ranking")
    graph.add_conditional_edges("ranking", route_after_ranking)
    graph.add_edge("explanation", "save_session")
    graph.add_edge("fallback", "save_session")
    graph.add_edge("save_session", END)
    return graph.compile()


shopping_graph = build_graph()


def run_shopping_graph(session_id: str, message: str):
    return shopping_graph.invoke({"session_id": session_id, "message": message})