"""End-to-end LangGraph agent tests driven by a scripted fake chat model.

These tests verify that:
- the agent routes the user's first turn through a tool of our choice,
- the tool's real observation is captured in the trace,
- the final AIMessage text is surfaced as `TurnResult.answer`.

No OpenAI key is needed.
"""

from __future__ import annotations

from travel_advisor.agent import build_agent, run_turn, run_turn_stream
from travel_advisor.tools import ALL_TOOLS

from .conftest import FakeToolCallingChatModel, make_final, make_tool_call


def test_run_turn_routes_to_rag_tool(installed_retriever):
    fake_llm = FakeToolCallingChatModel(
        replies=[
            make_tool_call(
                "search_travel_knowledge",
                {"query": "Hoi An ancient town", "top_k": 3},
            ),
            make_final("Hoi An is a riverside town in central Vietnam (dest-hoian)."),
        ]
    )
    agent = build_agent(fake_llm, tools=ALL_TOOLS)
    result = run_turn(agent, "Tell me about Hoi An.")
    assert "Hoi An" in result.answer
    assert len(result.trace) == 1
    assert result.trace[0].name == "search_travel_knowledge"
    # The observation comes from the real installed retriever — should be JSON-ish.
    assert "doc_id" in result.trace[0].observation


def test_run_turn_uses_sql_tool_for_flights():
    fake_llm = FakeToolCallingChatModel(
        replies=[
            make_tool_call(
                "query_tour_inventory",
                {
                    "table": "flights",
                    "filters": {"origin": "HAN", "destination": "PQC"},
                    "limit": 1,
                },
            ),
            make_final("Cheapest HAN→PQC option found."),
        ]
    )
    agent = build_agent(fake_llm, tools=ALL_TOOLS)
    result = run_turn(agent, "Cheapest flight HAN to PQC?")
    assert "Cheapest" in result.answer
    assert result.trace[0].name == "query_tour_inventory"
    assert "HAN" in result.trace[0].observation
    assert "PQC" in result.trace[0].observation


def test_run_turn_calls_budget_calculator():
    fake_llm = FakeToolCallingChatModel(
        replies=[
            make_tool_call(
                "estimate_trip_budget",
                {
                    "nights_hotel": 5,
                    "hotel_price_per_night_vnd": 1_000_000,
                    "travellers": 2,
                },
            ),
            make_final("Budget computed below."),
        ]
    )
    agent = build_agent(fake_llm, tools=ALL_TOOLS)
    result = run_turn(agent, "Budget for 2 people 5 nights at 1M/night?")
    assert result.trace[0].name == "estimate_trip_budget"
    assert "total_vnd" in result.trace[0].observation


def test_run_turn_empty_question_returns_blank():
    fake_llm = FakeToolCallingChatModel(replies=[make_final("never called")])
    agent = build_agent(fake_llm, tools=ALL_TOOLS)
    result = run_turn(agent, "   ")
    assert result.answer == ""
    assert result.trace == []


def test_run_turn_stream_emits_tool_and_answer_events(installed_retriever):
    fake_llm = FakeToolCallingChatModel(
        replies=[
            make_tool_call(
                "search_travel_knowledge",
                {"query": "Hoi An tailor shops", "top_k": 2},
            ),
            make_final("Hoi An has reputable tailors (dest-hoian)."),
        ]
    )
    agent = build_agent(fake_llm, tools=ALL_TOOLS)
    events = list(run_turn_stream(agent, "Where can I get a suit in Hoi An?"))
    kinds = [e.kind for e in events]
    # Expect at least one tool_call, one tool_result, one answer_chunk, one final.
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "answer_chunk" in kinds
    assert kinds[-1] == "final"
    # The final TurnResult must carry the full trace + answer.
    final = events[-1].data
    assert len(final.trace) == 1
    assert final.trace[0].name == "search_travel_knowledge"
    assert "Hoi An" in final.answer


def test_run_turn_stream_blank_question_yields_only_final():
    fake_llm = FakeToolCallingChatModel(replies=[make_final("never called")])
    agent = build_agent(fake_llm, tools=ALL_TOOLS)
    events = list(run_turn_stream(agent, "   "))
    assert len(events) == 1
    assert events[0].kind == "final"
    assert events[0].data.answer == ""


def test_run_turn_handles_multi_tool_sequence(installed_retriever):
    fake_llm = FakeToolCallingChatModel(
        replies=[
            make_tool_call(
                "search_travel_knowledge",
                {"query": "Sapa trek", "top_k": 2},
            ),
            make_tool_call(
                "query_tour_inventory",
                {"table": "tours", "filters": {"region": "north", "category": "trek"}, "limit": 2},
            ),
            make_final("Here are options + booking inventory."),
        ]
    )
    agent = build_agent(fake_llm, tools=ALL_TOOLS)
    result = run_turn(agent, "Plan a 2-day Sapa trek and show tours.")
    assert len(result.trace) == 2
    names = [t.name for t in result.trace]
    assert names == ["search_travel_knowledge", "query_tour_inventory"]
