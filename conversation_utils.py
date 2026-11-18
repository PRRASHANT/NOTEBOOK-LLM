"""Bound retained conversation turns for a document session."""

MAX_STORED_TURNS = 20


def append_conversation_turn(turns: list[dict], turn: dict) -> None:
    """Append a completed turn and discard the oldest turns beyond the limit.

    Mutates the session-state list in place so existing Streamlit references stay valid.
    """
    turns.append(turn)
    if len(turns) > MAX_STORED_TURNS:
        del turns[:-MAX_STORED_TURNS]
