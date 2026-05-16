"""Streamlit entry-point for the Vietnam Travel Planner chatbot.

Run with:

    streamlit run app.py

The app boots in three lazy stages so failures surface cleanly:

1. Sidebar input collects the OpenAI key (env var pre-fills it). The
   agent auto-boots once a non-empty key is present.
2. On boot, the FAISS index is loaded and the LangGraph agent is wired;
   both are cached in `st.session_state`.
3. Each user turn is streamed token-by-token via `run_turn_stream`, with
   tool calls rendered live inside an `st.status` panel.
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


# ---------------------------------------------------------------------------
# Page config & light styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vietnam Travel Planner",
    page_icon="🇻🇳",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; padding-bottom: 5rem; }
      [data-testid="stChatMessage"] { padding: 0.4rem 0.6rem; }
      .vt-hero { text-align: left; margin-bottom: 0.5rem; }
      .vt-hero h1 { margin: 0 0 0.25rem 0; font-size: 1.85rem; }
      .vt-hero p  { margin: 0; color: rgba(120,120,120,0.95); font-size: 0.95rem; }
      .vt-tool-name { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "chat_history": [],
    "thread_id": None,
    "agent": None,
    "retriever_loaded": False,
    "boot_error": None,
    "pending_question": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if st.session_state.thread_id is None:
    st.session_state.thread_id = str(uuid.uuid4())


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
    except Exception as exc:
        st.session_state.boot_error = str(exc)
        st.session_state.agent = None


def _format_args(args: dict) -> str:
    """Compact one-line summary of tool args for status display."""
    if not args:
        return ""
    return ", ".join(
        f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()
    )


def _render_tool_trace(trace: list[dict]) -> None:
    """Pretty-print the tool trace inside the current container."""
    if not trace:
        return
    label = f"🔧 {len(trace)} tool call" + ("s" if len(trace) > 1 else "")
    with st.expander(label, expanded=False):
        for i, call in enumerate(trace, 1):
            st.markdown(
                f"**{i}.** <span class='vt-tool-name'>{call['name']}</span>",
                unsafe_allow_html=True,
            )
            args_text = _format_args(call.get("args") or {})
            if args_text:
                st.caption(args_text)
            st.code(call["observation"], language="json", wrap_lines=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🇻🇳 Travel Planner")
    st.caption("RAG · LangGraph · FAISS")
    st.divider()

    api_key_default = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input(
        "OpenAI API key",
        value=api_key_default,
        type="password",
        help="Stored only in this browser session.",
    )

    # Auto-boot once a key is supplied; surface errors inline.
    if api_key and st.session_state.agent is None and not st.session_state.boot_error:
        with st.spinner("Booting agent..."):
            _boot_agent(api_key)

    if not api_key:
        st.warning("🔑 Add your OpenAI key to start.")
    elif st.session_state.boot_error:
        st.error(f"Boot failed: {st.session_state.boot_error}")
        if st.button("Retry boot", use_container_width=True):
            st.session_state.boot_error = None
            _boot_agent(api_key)
            st.rerun()
    elif st.session_state.agent is None:
        st.info("Booting…")
    else:
        st.success("✓ Agent ready")
        st.caption(f"Thread `{st.session_state.thread_id[:8]}`")

    st.divider()

    st.markdown("**Conversation**")
    if st.button("🔄 New conversation", use_container_width=True):
        _reset_conversation()
        st.rerun()

    st.divider()

    st.markdown("**Sample queries**")
    samples_path = Path(__file__).parent / "sample_queries.json"
    if samples_path.exists():
        scenarios = json.loads(samples_path.read_text(encoding="utf-8"))["scenarios"]
        groups: dict[str, list[dict]] = {}
        for s in scenarios:
            top = s["category"].split("—")[0].strip()
            groups.setdefault(top, []).append(s)
        agent_ready = st.session_state.agent is not None
        for category, items in groups.items():
            with st.expander(category, expanded=False):
                for s in items:
                    if st.button(
                        s["question"],
                        key=f"sample-{s['id']}",
                        use_container_width=True,
                        disabled=not agent_ready,
                        help=s["question"],
                    ):
                        st.session_state.pending_question = s["question"]
                        st.rerun()


# ---------------------------------------------------------------------------
# Main: hero + chat
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="vt-hero">
      <h1>Vietnam Travel Planner</h1>
      <p>Hỏi gì cũng được — điểm đến, lịch trình, tour, khách sạn, chuyến bay, visa, ngân sách.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Empty-state hint only on a fresh conversation.
if not st.session_state.chat_history and st.session_state.agent is not None:
    with st.container(border=True):
        st.markdown(
            "👋 **Bắt đầu nào!** Thử một câu hỏi mẫu ở thanh bên, hoặc gõ câu hỏi "
            "của bạn vào ô bên dưới — trợ lý sẽ truy xuất kiến thức, tra cứu "
            "tour/khách sạn và tính ngân sách giúp bạn."
        )
elif not st.session_state.chat_history and st.session_state.agent is None:
    with st.container(border=True):
        st.markdown(
            "⚙️ Nhập **OpenAI API key** ở thanh bên trái để khởi động trợ lý."
        )

# Render conversation history (no in-stream rendering for past turns).
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("trace"):
            _render_tool_trace(turn["trace"])

# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

pending = st.session_state.pending_question
if pending:
    st.session_state.pending_question = None

agent_ready = st.session_state.agent is not None
prompt = pending or st.chat_input(
    "Hỏi về điểm đến, tour, khách sạn, chuyến bay, ngân sách…",
    disabled=not agent_ready,
)

if prompt:
    if not agent_ready:
        st.error("Trợ lý chưa sẵn sàng — thêm OpenAI key ở thanh bên trước.")
    else:
        # 1) Render the user message immediately.
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2) Stream the assistant response.
        with st.chat_message("assistant"):
            status = st.status("🤔 Đang suy nghĩ…", expanded=True)
            answer_box = st.empty()
            accumulated: list[str] = []
            final_trace: list[dict] = []
            error_text: str | None = None

            try:
                for event in run_turn_stream(
                    st.session_state.agent,
                    prompt,
                    thread_id=st.session_state.thread_id,
                ):
                    if event.kind == "tool_call":
                        name = event.data["name"]
                        args_text = _format_args(event.data.get("args") or {})
                        status.update(label=f"🔧 Đang gọi `{name}`…")
                        bullet = f"**→ `{name}`**"
                        if args_text:
                            bullet += f" · `{args_text}`"
                        status.markdown(bullet)
                    elif event.kind == "tool_result":
                        status.markdown(f"**✓ `{event.data['name']}` xong**")
                    elif event.kind == "answer_chunk":
                        accumulated.append(event.data)
                        answer_box.markdown("".join(accumulated) + " ▌")
                    elif event.kind == "final":
                        final_trace = [t.to_dict() for t in event.data.trace]
            except Exception as exc:
                error_text = f"⚠️ Lỗi: `{exc}`"

            # Settle the UI based on outcome.
            if error_text:
                status.update(label="❌ Có lỗi xảy ra", state="error", expanded=True)
                answer_box.error(error_text)
                final_answer = error_text
            else:
                n = len(final_trace)
                status.update(
                    label=f"✓ Xong ({n} tool call{'s' if n != 1 else ''})",
                    state="complete",
                    expanded=False,
                )
                final_answer = "".join(accumulated).strip() or "_(empty response)_"
                answer_box.markdown(final_answer)
                _render_tool_trace(final_trace)

        # 3) Persist the turn so it shows up on the next rerun.
        st.session_state.chat_history.append(
            {"role": "assistant", "content": final_answer, "trace": final_trace}
        )
