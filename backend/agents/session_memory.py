SESSIONS = {}


def load_session(state):
    session = SESSIONS.get(state["session_id"], {})
    return {
        "constraints": session.get("constraints", {}),
        "history": session.get("history", []),
        "turn": session.get("turn", 0),
        "pending_clarification": session.get("pending_clarification"),
        "pending_clarification_question": session.get("pending_clarification_question"),
        "previous_products": session.get("recommended_products", []),
    }


def save_session(state):
    history = list(state.get("history", []))
    history.append({
        "role": "user",
        "message": state.get("message", ""),
        "constraints": state.get("constraints", {}),
    })
    history.append({
        "role": "assistant",
        "action": state.get("action", "message"),
        "message": state.get("response_message", ""),
        "search_status": state.get("search_status", "success"),
        "fallback_suggestions": state.get("fallback_suggestions", []),
    })

    current_recs = state.get("recommended_products") or state.get("ranked_products") or state.get("candidates")
    if current_recs is None:
        current_recs = state.get("previous_products", [])

    SESSIONS[state["session_id"]] = {
        "constraints": state.get("constraints", {}),
        "history": history[-20:],
        "turn": state.get("turn", 1),
        "pending_clarification": state.get("pending_clarification"),
        "pending_clarification_question": state.get("clarifying_question"),
        "recommended_products": current_recs or [],
    }

    return {}
