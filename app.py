"""Streamlit entry-point for the Vietnam Travel Planner chatbot.

Run with:

    streamlit run app.py

The app boots in three lazy stages so failures surface cleanly:

1. Sidebar input collects the OpenAI key (env var pre-fills it).
2. On first message, the FAISS index is loaded from disk and the
   LangGraph agent is constructed; both are cached in `st.session_state`.
3. Each user turn is sent through `run_turn(...)` and rendered with a
   citations expander + a tool-call trace expander.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from travel_advisor.agent import build_agent, build_llm, run_turn_stream
from travel_advisor.config import settings
from travel_advisor.retriever import Retriever, set_retriever


st.set_page_config(
    page_title="Vietnam Travel Planner",
    page_icon="🇻🇳",
    layout="wide",
)

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[dict(role, content, trace?)]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent" not in st.session_state:
    st.session_state.agent = None
if "retriever_loaded" not in st.session_state:
    st.session_state.retriever_loaded = False
if "boot_error" not in st.session_state:
    st.session_state.boot_error = None


def _reset_conversation() -> None:
    st.session_state.chat_history = []
    st.session_state.thread_id = str(uuid.uuid4())


def _boot_agent(api_key: str) -> None:
    """Idempotent boot: load FAISS retriever + OpenAI agent into session state."""
    try:
        if not st.session_state.retriever_loaded:
            from travel_advisor.embeddings import OpenAIEmbedder

            embedder = OpenAIEmbedder(api_key=api_key)
            retriever = Retriever.load(settings.index_dir, embedder)
            set_retriever(retriever)
            st.session_state.retriever_loaded = True
        if st.session_state.agent is None:
            llm = build_llm(api_key=api_key)
            st.session_state.agent = build_agent(llm)
        st.session_state.boot_error = None
    except Exception as exc:  # surface to UI rather than crash the page
        st.session_state.boot_error = str(exc)
        st.session_state.agent = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🇻🇳 Vietnam Travel Planner")
    st.caption("RAG + LangGraph + FAISS demo")

    api_key_default = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API key",
        value=api_key_default,
        type="password",
        help="Your OpenAI API key. Stored only in this session.",
    )

    if st.button("Boot / Reload agent", disabled=not api_key):
        _boot_agent(api_key)

    st.divider()
    st.subheader("Status")
    if not api_key:
        st.warning("Enter your OpenAI key to start.")
    elif st.session_state.boot_error:
        st.error(f"Boot error: {st.session_state.boot_error}")
    elif st.session_state.agent is None:
        st.info("Click **Boot / Reload agent** to load FAISS + LLM.")
    else:
        st.success("Agent ready.")
        st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")

    st.divider()
    if st.button("New conversation", use_container_width=True):
        _reset_conversation()
        st.rerun()

    st.divider()
    st.subheader("Sample queries")
    samples_path = Path(__file__).parent / "sample_queries.json"
    if samples_path.exists():
        scenarios = json.loads(samples_path.read_text(encoding="utf-8"))["scenarios"]
        for s in scenarios:
            if st.button(
                f"{s['id']} — {s['category']}",
                key=f"sample-{s['id']}",
                use_container_width=True,
                help=s["question"],
            ):
                st.session_state.pending_question = s["question"]
                st.rerun()


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------

st.markdown("### Chat")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("trace"):
            with st.expander(f"🔧 Tool calls ({len(turn['trace'])})", expanded=False):
                for i, call in enumerate(turn["trace"], 1):
                    st.markdown(f"**{i}. `{call['name']}`** — args:")
                    st.code(json.dumps(call["args"], ensure_ascii=False, indent=2), language="json")
                    st.markdown("Observation:")
                    st.code(call["observation"], language="json")

# Use pending_question if a sample was clicked.
pending = st.session_state.pop("pending_question", None)
prompt = pending or st.chat_input(
    "Ask about destinations, transport, visa, food, tours, hotels..."
)

if prompt:
    if not api_key:
        st.error("Add your OpenAI key in the sidebar first.")
    else:
        if st.session_state.agent is None:
            _boot_agent(api_key)
        if st.session_state.boot_error:
            st.error(f"Boot error: {st.session_state.boot_error}")
        elif st.session_state.agent is not None:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                status_box = st.empty()
                answer_box = st.empty()
                trace_payload: list[dict] = []
                answer_buffer: list[str] = []
                error: str | None = None
                try:
                    for event in run_turn_stream(
                        st.session_state.agent,
                        prompt,
                        thread_id=st.session_state.thread_id,
                    ):
                        if event.kind == "tool_call":
                            status_box.info(
                                f"🔧 Calling `{event.data['name']}`..."
                            )
                        elif event.kind == "tool_result":
                            status_box.info(
                                f"✅ `{event.data['name']}` finished"
                            )
                        elif event.kind == "answer_chunk":
                            answer_buffer.append(event.data)
                            answer_box.markdown("".join(answer_buffer))
                        elif event.kind == "final":
                            trace_payload = [t.to_dict() for t in event.data.trace]
                            if event.data.answer:
                                answer_box.markdown(event.data.answer)
                                answer_buffer = [event.data.answer]
                except Exception as exc:
                    error = f"⚠️ Agent error: `{exc}`"
                    answer_box.markdown(error)
                status_box.empty()
                if trace_payload:
                    with st.expander(
                        f"🔧 Tool calls ({len(trace_payload)})", expanded=False
                    ):
                        for i, call in enumerate(trace_payload, 1):
                            st.markdown(f"**{i}. `{call['name']}`** — args:")
                            st.code(
                                json.dumps(call["args"], ensure_ascii=False, indent=2),
                                language="json",
                            )
                            st.markdown("Observation:")
                            st.code(call["observation"], language="json")
            final_answer = error or ("".join(answer_buffer) or "_(empty response)_")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": final_answer, "trace": trace_payload}
            )
