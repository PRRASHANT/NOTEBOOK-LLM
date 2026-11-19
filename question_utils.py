"""Validate document questions before retrieval or model calls."""

MAX_QUESTION_CHARACTERS = 2000


def validate_question(question: str) -> str:
    """Return a trimmed question or reject blank and oversized input."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Enter a question to search the document.")
    if len(cleaned) > MAX_QUESTION_CHARACTERS:
        raise ValueError(f"Question exceeds the {MAX_QUESTION_CHARACTERS}-character limit.")
    return cleaned
