"""LangGraph ReAct agent wired to the four travel-advisor tools.

Exposes:
- `build_llm()`  — production AzureChatOpenAI factory.
- `build_agent(llm, tools=...)` — wraps LLM + tools into a graph with a
  `MemorySaver` checkpointer so multi-turn threads remember context.
- `run_turn(...)` — high-level helper: takes a question, returns the
  final answer plus a structured tool trace for UI display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator
import os
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .config import settings
from .tools import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are a helpful Vietnam Travel Planner assistant. You help foreign "
    "and Vietnamese travellers plan trips to Vietnam: destinations, "
    "transport, accommodation, food, visa, safety, and budgeting.\n\n"
    "Reply in the same language the user used (Vietnamese if Vietnamese, "
    "English if English).\n\n"
    "Tool usage rules:\n"
    "  - For ANY question about destinations, transport, food, visa, "
    "    safety, accommodation, sample itineraries — call "
    "    `search_travel_knowledge` FIRST and ground your answer in the "
    "    returned snippets. Cite each fact you use with (doc_id), e.g. "
    "    (dest-hoian) or (visa-evisa).\n"
    "  - For real-time inventory / availability / specific prices / "
    "    booking options (tours, hotels, flights), call "
    "    `query_tour_inventory` with the appropriate table and filters.\n"
    "  - To total up a trip's cost, ALWAYS call `estimate_trip_budget` "
    "    rather than computing in your head — that way the user sees the "
    "    breakdown.\n"
    "  - For time-sensitive web information (weather alerts, recent news), "
    "    call `web_search_news`. If it returns mode='offline', tell the "
    "    user that live data isn't available in this demo.\n"
    "  - You may call multiple tools in one turn when the user combines "
    "    questions.\n\n"
    "Answering rules:\n"
    "  - Quote prices, durations, and policy details exactly as returned "
    "    by the tools. Never invent numbers.\n"
    "  - If retrieval returns nothing relevant, say so explicitly — do "
    "    not pretend you have an answer.\n"
    "  - If the user asks about something outside Vietnam travel (stocks, "
    "    coding help, etc.), politely steer back to travel planning.\n"
    "  - Keep answers concise: short prose for facts, bullet lists for "
    "    options and itineraries.\n"
)


@dataclass
class ToolTrace:
    name: str
    args: dict[str, Any]
    observation: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args, "observation": self.observation}


@dataclass
class TurnResult:
    answer: str
    trace: list[ToolTrace] = field(default_factory=list)
    thread_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "trace": [t.to_dict() for t in self.trace],
            "thread_id": self.thread_id,
        }


def build_llm(
    *,
    api_key: str | None = None,
    deployment: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 1024,
) -> AzureChatOpenAI:
    """Construct the Azure-hosted chat model used by the agent."""
    key = api_key or settings.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "AZURE_OPENAI_API_KEY missing — copy .env.example to .env "
            "and paste the key from the course staff."
        )
    return AzureChatOpenAI(
        azure_endpoint=settings.endpoint,
        api_version=settings.api_version,
        azure_deployment=deployment or settings.chat_deployment,
        api_key=key,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=max_tokens,
    )


def build_agent(
    llm: Any,
    tools: list[Any] | None = None,
    *,
    checkpointer: Any | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Any:
    """Wire LLM + tools into a LangGraph ReAct agent with a checkpointer."""
    return create_react_agent(
        llm,
        list(tools) if tools is not None else list(ALL_TOOLS),
        prompt=system_prompt,
        checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
    )


def _config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.recursion_limit,
    }


def _extract_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                parts.append(chunk.get("text", ""))
            elif isinstance(chunk, str):
                parts.append(chunk)
        return "\n".join(p for p in parts if p)
    return ""


def _collect_trace(messages: list[Any]) -> tuple[str, list[ToolTrace]]:
    """Walk the agent's message log, return (final answer, tool trace)."""
    trace: list[ToolTrace] = []
    pending: dict[str, dict[str, Any]] = {}
    final_text = ""

    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for call in tool_calls:
                call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
                if call_id:
                    pending[call_id] = {"name": name, "args": args or {}}
            text = _extract_text(msg)
            if text:
                final_text = text
        elif msg.__class__.__name__ == "ToolMessage":
            call_id = getattr(msg, "tool_call_id", None)
            content = getattr(msg, "content", "")
            if call_id and call_id in pending:
                spec = pending.pop(call_id)
                trace.append(
                    ToolTrace(
                        name=spec["name"],
                        args=spec["args"],
                        observation=str(content),
                    )
                )
            else:
                trace.append(
                    ToolTrace(name="?", args={}, observation=str(content))
                )
    return final_text, trace


def run_turn(
    agent: Any,
    question: str,
    *,
    thread_id: str | None = None,
) -> TurnResult:
    """Single agent turn. Returns the final answer + tool trace."""
    if not question or not question.strip():
        return TurnResult(answer="", trace=[], thread_id=thread_id or "")
    tid = thread_id or str(uuid.uuid4())
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=_config(tid),
    )
    messages = result.get("messages", []) if isinstance(result, dict) else []
    answer, trace = _collect_trace(messages)
    return TurnResult(answer=answer, trace=trace, thread_id=tid)


@dataclass
class StreamEvent:
    """A single event surfaced from the agent's streaming loop.

    `kind` is one of:
      - "tool_call"     — LLM requested a tool. `data` = {"name", "args"}.
      - "tool_result"   — tool finished. `data` = {"name", "observation"}.
      - "answer_chunk"  — incremental piece of the final answer text.
      - "final"         — last event, carries the full TurnResult.
    """

    kind: str
    data: Any


def run_turn_stream(
    agent: Any,
    question: str,
    *,
    thread_id: str | None = None,
) -> Iterator[StreamEvent]:
    """Stream events from the agent as it works.

    Yields tool-call and tool-result events live, then the assistant's
    answer text in incremental chunks, then a single `final` event with
    the full `TurnResult`. UI layers can render each event as it arrives.

    Falls back to one-shot mode if the agent does not expose `.stream()`
    (e.g. the fake LLM used in tests).
    """
    if not question or not question.strip():
        yield StreamEvent(
            kind="final",
            data=TurnResult(answer="", trace=[], thread_id=thread_id or ""),
        )
        return
    tid = thread_id or str(uuid.uuid4())

    stream_fn = getattr(agent, "stream", None)
    if stream_fn is None:
        result = run_turn(agent, question, thread_id=tid)
        if result.answer:
            yield StreamEvent(kind="answer_chunk", data=result.answer)
        yield StreamEvent(kind="final", data=result)
        return

    seen_tool_call_ids: set[str] = set()
    seen_tool_message_ids: set[str] = set()
    pending: dict[str, dict[str, Any]] = {}
    trace: list[ToolTrace] = []
    answer_text = ""
    emitted_text = ""

    for step in stream_fn(
        {"messages": [HumanMessage(content=question)]},
        config=_config(tid),
        stream_mode="values",
    ):
        messages = step.get("messages", []) if isinstance(step, dict) else []
        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in getattr(msg, "tool_calls", None) or []:
                    call_id = (
                        call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                    )
                    if not call_id or call_id in seen_tool_call_ids:
                        continue
                    seen_tool_call_ids.add(call_id)
                    name = (
                        call.get("name") if isinstance(call, dict)
                        else getattr(call, "name", "?")
                    )
                    args = (
                        call.get("args") if isinstance(call, dict)
                        else getattr(call, "args", {})
                    ) or {}
                    pending[call_id] = {"name": name, "args": args}
                    yield StreamEvent(
                        kind="tool_call",
                        data={"name": name, "args": args},
                    )
                text = _extract_text(msg)
                if text:
                    answer_text = text
            elif isinstance(msg, ToolMessage):
                msg_id = getattr(msg, "id", None) or getattr(msg, "tool_call_id", None)
                if msg_id and msg_id in seen_tool_message_ids:
                    continue
                if msg_id:
                    seen_tool_message_ids.add(msg_id)
                call_id = getattr(msg, "tool_call_id", None)
                content = str(getattr(msg, "content", ""))
                spec = pending.pop(call_id, None) if call_id else None
                name = spec["name"] if spec else "?"
                args = spec["args"] if spec else {}
                trace.append(ToolTrace(name=name, args=args, observation=content))
                yield StreamEvent(
                    kind="tool_result",
                    data={"name": name, "observation": content},
                )

        # Emit any new tail of the answer as a chunk.
        if answer_text and answer_text != emitted_text:
            if answer_text.startswith(emitted_text):
                delta = answer_text[len(emitted_text):]
            else:
                delta = answer_text  # text was rewritten; resend in full
            if delta:
                yield StreamEvent(kind="answer_chunk", data=delta)
            emitted_text = answer_text

    yield StreamEvent(
        kind="final",
        data=TurnResult(answer=answer_text, trace=trace, thread_id=tid),
    )
