SESSIONS = {}


def load_session(state):

    session = SESSIONS.get(state["session_id"], {})

    return {
        "constraints": session.get("constraints", {}),
        "history": session.get("history", []),
        "turn": session.get("turn", 0)
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
    })
    SESSIONS[state["session_id"]] = {
        "constraints": state.get("constraints", {}),
        "history": history[-20:],
        "turn": state.get("turn", 1),
    }

    return {}