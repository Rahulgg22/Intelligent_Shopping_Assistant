import logging
import re
import time

from langgraph.graph import END, START, StateGraph

from backend.agents.attribute_mapping import map_attributes
from backend.agents.clarification import check_clarification
from backend.agents.explanation import explain_recommendations
from backend.agents.fallback import handle_fallback
from backend.agents.intent_extraction import extract_intent
from backend.agents.query_ingestion import ingest
from backend.agents.ranking import rank_candidates
from backend.agents.session_memory import load_session, save_session
from backend.graph.state import ShoppingState
from backend.services.mcp_client import get_mcp_client
from backend.services.product_sources import build_product_service

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
    if state.get("needs_clarification"):
        return "clarification"
    if state.get("tool_intent") in ("compare", "reviews", "alternatives", "explain_ranking"):
        return "mcp_tool_execution"
    return "mapping"


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
    query = filters.get("search_query") or state.get("normalized_query", state.get("message", ""))
    result = build_product_service().search(
        query,
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


def mcp_tool_execution_node(state):
    """
    Executes MCP tools through the central MCPClient based on conversational planner intent.
    Supports compare_products, search_product_reviews, and find_product_alternatives.
    Handles errors gracefully so tool failures never crash the workflow.
    """
    tool_intent = state.get("tool_intent")
    client = get_mcp_client()
    previous = state.get("previous_products") or []
    constraints = state.get("constraints") or {}
    query = state.get("normalized_query", state.get("message", ""))

    try:
        if tool_intent == "compare":
            count = 3
            if re.search(r"\b(?:two|both|top\s*2|2)\b", query, re.I):
                count = 2
            elif re.search(r"\b(?:three|top\s*3|3)\b", query, re.I):
                count = 3
            target_products = previous[:count]
            user_reqs = list(constraints.get("preferences", []))
            if constraints.get("budget_max"):
                user_reqs.append(f"Under ₹{int(constraints['budget_max']):,}")
            if constraints.get("intended_use"):
                user_reqs.append(f"Suitable for {constraints['intended_use']}")

            comp_result = client.call_tool("compare_products", {
                "products": target_products,
                "user_requirements": user_reqs,
            })

            msg_parts = [
                f"Here is a side-by-side comparison of the {len(target_products)} selected products:\n"
            ]
            if comp_result.get("summary"):
                msg_parts.append(f"**Key Takeaway**: {comp_result['summary']}\n")
            if comp_result.get("trade_offs"):
                msg_parts.append("**Trade-offs & Differences**:\n- " + "\n- ".join(comp_result["trade_offs"]))
            if comp_result.get("common_features"):
                msg_parts.append("\n**Common Features**:\n- " + "\n- ".join(comp_result["common_features"]))
            if comp_result.get("missing_information"):
                msg_parts.append("\n**Notes / Unverified Details**:\n- " + "\n- ".join(comp_result["missing_information"][:3]))

            updated_products = []
            for i, p in enumerate(target_products):
                p_copy = dict(p)
                comp_item = comp_result.get("comparison", [])[i] if i < len(comp_result.get("comparison", [])) else {}
                p_copy["matched_requirements"] = comp_item.get("matched_requirements", p_copy.get("matched_requirements", []))
                p_copy["benefits"] = comp_item.get("advantages", p_copy.get("benefits", []))
                p_copy["limitations"] = comp_item.get("disadvantages", p_copy.get("limitations", []))
                p_copy["comparison_note"] = comp_item.get("advantages", [""])[0] if comp_item.get("advantages") else None
                updated_products.append(p_copy)

            return {
                "action": "recommendations",
                "response_message": "\n".join(msg_parts),
                "recommended_products": updated_products,
                "comparison_result": comp_result,
                "selected_tool": "compare_products",
            }

        elif tool_intent == "reviews":
            target_prod = previous[0] if previous else {}
            if len(previous) >= 2 and re.search(r"\b(?:second|2nd)\b", query, re.I):
                target_prod = previous[1]
            elif len(previous) >= 3 and re.search(r"\b(?:third|3rd)\b", query, re.I):
                target_prod = previous[2]

            prod_name = target_prod.get("name") or target_prod.get("title") or "product"
            brand = target_prod.get("brand") or ""
            search_ctx = constraints.get("intended_use") or " ".join(constraints.get("preferences", []))

            rev_result = client.call_tool("search_product_reviews", {
                "product_name": prod_name,
                "brand": brand,
                "search_context": search_ctx,
                "max_results": 6,
            })

            msg_parts = [f"### Independent Reviews & Feedback for {prod_name}\n"]
            if rev_result.get("common_positive_points"):
                msg_parts.append("**What Reviewers Liked (Pros)**:")
                for pt in rev_result["common_positive_points"]:
                    msg_parts.append(f"- ✓ {pt}")
                msg_parts.append("")
            if rev_result.get("common_negative_points"):
                msg_parts.append("**Common Drawbacks & Complaints (Cons)**:")
                for pt in rev_result["common_negative_points"]:
                    msg_parts.append(f"- ⚠ {pt}")
                msg_parts.append("")
            if rev_result.get("sources"):
                msg_parts.append(f"**Sources Scanned**: {', '.join(rev_result['sources'][:5])}")

            if not rev_result.get("common_positive_points") and not rev_result.get("common_negative_points"):
                msg_parts.append(rev_result.get("limitations", ["No verified independent reviews found."])[0])

            return {
                "action": "message",
                "response_message": "\n".join(msg_parts),
                "recommended_products": [target_prod] if target_prod else [],
                "review_result": rev_result,
                "selected_tool": "search_product_reviews",
            }

        elif tool_intent == "alternatives":
            target_prod = previous[0] if previous else {}
            alt_type = state.get("alternative_type") or "similar"

            user_reqs = {
                "max_price": constraints.get("budget_max"),
                "preferred_brands": constraints.get("preferred_brands", []),
                "excluded_brands": constraints.get("excluded_brands", []),
                "required_features": constraints.get("preferences", []),
            }

            alt_result = client.call_tool("find_product_alternatives", {
                "original_product": target_prod,
                "alternative_type": alt_type,
                "user_requirements": user_reqs,
                "max_results": 8,
            })

            alternatives = alt_result.get("alternatives", [])
            if alternatives:
                type_label = alt_type.replace("_", " ")
                msg = f"Here are verified {type_label} alternatives to **{target_prod.get('name', 'your selected product')}** based on your preferences."
                if alt_result.get("notes"):
                    msg += f" ({'; '.join(alt_result['notes'])})"
                return {
                    "action": "alternatives",
                    "response_message": msg,
                    "recommended_products": alternatives,
                    "alternatives": alternatives,
                    "alternatives_result": alt_result,
                    "selected_tool": "find_product_alternatives",
                }
            else:
                notes_str = "; ".join(alt_result.get("notes", ["No matching options discovered."]))
                return {
                    "action": "message",
                    "response_message": f"No {alt_type.replace('_', ' ')} alternatives could be found for **{target_prod.get('name', 'this product')}** under the current constraints. {notes_str}",
                    "recommended_products": [],
                    "alternatives": [],
                    "alternatives_result": alt_result,
                    "selected_tool": "find_product_alternatives",
                }

        elif tool_intent == "explain_ranking":
            from backend.agents.explanation import handle_explanation_followup
            return handle_explanation_followup(query, previous, constraints)

        return {"action": "message", "response_message": "Unsupported tool request."}
    except Exception as exc:
        timing_logger.error("mcp_tool_execution encountered error: %s", exc)
        return {
            "action": "message",
            "response_message": f"I encountered an issue processing your {tool_intent} request. Please try again or rephrase.",
            "recommended_products": previous,
            "selected_tool": f"{tool_intent}_error",
        }


def build_graph():
    graph = StateGraph(ShoppingState)
    nodes = {
        "load_session": load_session, "ingest": ingest, "extract_intent": extract_intent,
        "check_clarification": check_clarification, "clarification": clarification_node,
        "mcp_tool_execution": mcp_tool_execution_node,
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
    graph.add_edge("mcp_tool_execution", "save_session")
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
