import re


def ingest(state):

    text = state["message"]

    text = re.sub(
        r"\b(hey|hello|hi|please|um|uh)\b",
        "",
        text,
        flags=re.I
    )

    text = re.sub(r"\s+", " ", text).strip()

    return {
        "normalized_query": text,
        "turn": state.get("turn", 0) + 1
    }