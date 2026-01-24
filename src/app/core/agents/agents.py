"""Agent implementations for the multi-agent RAG flow."""

from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..llm.factory import create_chat_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


# =========================
# AGENT DEFINITIONS
# =========================

retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)


# =========================
# FEATURE 2: ENHANCED RETRIEVAL NODE
# =========================

def retrieval_node(state: QAState) -> QAState:
    question = state["question"]

    result = retrieval_agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    messages = result.get("messages", [])
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    raw_context_blocks: List[str] = []
    retrieval_logs: List[str] = []

    for i, tool_msg in enumerate(tool_messages):
        query_used = "Unknown Query"

        # try to match tool call with AIMessage
        if i < len(ai_messages):
            tool_calls = getattr(ai_messages[i], "tool_calls", None)
            if tool_calls:
                query_used = tool_calls[0].get("args", {}).get("query", "Unknown")

        context_block = str(tool_msg.content)
        raw_context_blocks.append(context_block)

        retrieval_logs.append(
            f"Retrieval Call {i + 1}\n"
            f"Query: {query_used}\n"
            f"Context Length: {len(context_block)} characters"
        )

    structured_context = ""
    for i, block in enumerate(raw_context_blocks):
        structured_context += f"\n=== RETRIEVAL CALL {i + 1} ===\n{block}\n"

    return {
        "context": structured_context.strip(),
        "raw_context_blocks": raw_context_blocks,
        "retrieval_traces": "\n\n".join(retrieval_logs),
    }


def summarization_node(state: QAState) -> QAState:
    question = state["question"]
    context = state.get("context", "")

    user_content = f"Question:\n{question}\n\nContext:\n{context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )

    draft_answer = _extract_last_ai_content(result.get("messages", []))

    return {"draft_answer": draft_answer}


def verification_node(state: QAState) -> QAState:
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""
Question:
{question}

Context:
{context}

Draft Answer:
{draft_answer}

Verify the answer strictly against the context.
"""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )

    answer = _extract_last_ai_content(result.get("messages", []))

    return {"answer": answer}
