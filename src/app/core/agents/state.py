"""LangGraph state schema for the multi-agent QA flow."""

from typing import List, TypedDict


class QAState(TypedDict):
    """State schema for the linear multi-agent QA flow.

    The state flows through three agents:
    1. Retrieval Agent: populates `context` from `question`
    2. Summarization Agent: generates `draft_answer` from `question` + `context`
    3. Verification Agent: produces final `answer` from `question` + `context` + `draft_answer`
    """

    question: str
    context: str | None
    draft_answer: str | None
    answer: str | None

    # --- NEW FIELDS FOR FEATURE 2 ---
    retrieval_traces: str | None          # Human-readable log of all search attempts
    raw_context_blocks: List[str] | None  # Individual context chunks from each call
