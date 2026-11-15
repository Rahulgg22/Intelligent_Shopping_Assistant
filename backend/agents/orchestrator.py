from backend.graph.workflow import run_shopping_graph


def process_message(session_id: str, message: str):
    return run_shopping_graph(session_id, message)