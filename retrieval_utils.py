"""Build document search queries that retain context for brief follow-ups."""

import re
from collections.abc import Sequence


_REFERENCE_WORDS = re.compile(r"\b(?:it|its|they|them|their|this|that|these|those)\b", re.IGNORECASE)
_FOLLOWUP_OPENING = re.compile(
    r"^(?:and\b|also\b|what about\b|how about\b|tell me more\b|"
    r"can you (?:elaborate|explain more|give (?:me )?an example)\b)",
    re.IGNORECASE,
)
MAX_CONTEXT_TURNS = 3
MAX_ANCHOR_LENGTH = 500


def _needs_context(question: str) -> bool:
    return bool(_REFERENCE_WORDS.search(question) or _FOLLOWUP_OPENING.search(question))


def build_retrieval_query(question: str, previous_questions: Sequence[str] = ()) -> str:
    """Include a prior self-contained question when a follow-up needs context.

    Only user questions are used for retrieval; generated answers are not evidence.
    Standalone questions remain unchanged to avoid pulling in unrelated topics.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Search question must not be blank.")
    current = question.strip()
    if not previous_questions or not _needs_context(current):
        return current

    recent = previous_questions[-MAX_CONTEXT_TURNS:]
    if any(not isinstance(item, str) or not item.strip() for item in recent):
        raise ValueError("Previous questions must contain nonempty text.")

    # Prefer the most recent standalone topic over an ambiguous follow-up.
    anchor = next(
        (item for item in reversed(recent) if not _needs_context(item)),
        recent[-1],
    )
    return f"{anchor.strip()[:MAX_ANCHOR_LENGTH]}\nFollow-up question: {current}"
