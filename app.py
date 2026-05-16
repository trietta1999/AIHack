"""Streamlit entry-point for the Vietnam Travel Planner chatbot.

Run with:

    streamlit run app.py

The app boots in three lazy stages so failures surface cleanly:

1. Sidebar input collects the OpenAI key (env var pre-fills it). The
   agent auto-boots once a non-empty key is present.
2. On boot, the FAISS index is loaded and the LangGraph agent is wired;
   both are cached in `st.session_state`.
3. Each user turn is streamed token-by-token via `run_turn_stream`, with
   tool calls rendered live inside an `st.status` panel. Messages
   persist to `data/chats.sqlite`, and the active thread id rides in the
   URL (`?t=<uuid>`) so reloads and shared links restore the same view.
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
from travel_advisor.chat_store import (
    delete_thread as store_delete_thread,
    init_store,
    list_threads,
    load_thread,
    save_turn,
)
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

      /* Sidebar conversation list — quiet rows, subtle active-state. */
      section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 0.35rem; }
      section[data-testid="stSidebar"] button {
        border-radius: 8px;
        font-weight: 400;
      }
      section[data-testid="stSidebar"] button[kind="secondary"] {
        background: transparent;
        border: 1px solid rgba(125,125,125,0.18);
      }
      section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(125,125,125,0.10);
        border-color: rgba(125,125,125,0.30);
      }
      section[data-testid="stSidebar"] button[kind="primary"] {
        background: rgba(99,102,241,0.14);
        color: inherit;
        border: 1px solid rgba(99,102,241,0.35);
        font-weight: 500;
      }
      section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: rgba(99,102,241,0.20);
      }
      /* Single-line titles for thread buttons; restored inside expanders
         so sample-query questions can still wrap. */
      section[data-testid="stSidebar"] button p {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin: 0;
      }
      section[data-testid="stSidebar"] [data-testid="stExpander"] button p {
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
      }

      .vt-section-label {
        margin: 0.25rem 0 0.5rem 0;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(125,125,125,0.95);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _truncate(text: str, max_len: int) -> str:
    """Trim long titles for the sidebar list."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def _format_when(iso_ts: str) -> str:
    """Compact display of an ISO timestamp: YYYY-MM-DD HH:MM."""
    if not iso_ts:
        return ""
    return iso_ts.replace("T", " ").split("+")[0][:16]


# ---------------------------------------------------------------------------
# Persistence bootstrap
# ---------------------------------------------------------------------------

init_store(settings.chats_db_path)


# ---------------------------------------------------------------------------
# Session state + thread resolution
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "chat_history": None,
    "thread_id": None,
    "agent": None,
    "retriever_loaded": False,
    "boot_error": None,
    "pending_question": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_qp_thread = st.query_params.get("t")
if st.session_state.thread_id is None:
    st.session_state.thread_id = _qp_thread or str(uuid.uuid4())
    st.session_state.chat_history = None
elif _qp_thread and _qp_thread != st.session_state.thread_id:
    # URL changed (shared link, back/forward); follow it and reload history.
    st.session_state.thread_id = _qp_thread
    st.session_state.chat_history = None

# Keep the URL in sync with the active thread.
if st.query_params.get("t") != st.session_state.thread_id:
    st.query_params["t"] = st.session_state.thread_id

# Lazily hydrate history from disk on first load or after a thread switch.
if st.session_state.chat_history is None:
    st.session_state.chat_history = load_thread(
        settings.chats_db_path, st.session_state.thread_id
    )


def _start_new_thread() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.query_params["t"] = st.session_state.thread_id


def _switch_thread(thread_id: str) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.chat_history = None  # forces hydrate on next pass
    st.query_params["t"] = thread_id


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
    if not args:
        return ""
    return ", ".join(
        f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()
    )


def _render_tool_trace(trace: list[dict]) -> None:
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

    st.markdown("<div class='vt-section-label'>Hội thoại</div>", unsafe_allow_html=True)
    if st.button("＋ Hội thoại mới", use_container_width=True, type="secondary"):
        _start_new_thread()
        st.rerun()

    past = list_threads(settings.chats_db_path)
    if past:
        st.markdown(
            f"<div class='vt-section-label' style='margin-top:1rem'>Đã lưu · {len(past)}</div>",
            unsafe_allow_html=True,
        )
        for t in past:
            is_current = t["thread_id"] == st.session_state.thread_id
            cols = st.columns([5, 1], gap="small", vertical_alignment="center")
            with cols[0]:
                if st.button(
                    _truncate(t["title"], 36),
                    key=f"open-{t['thread_id']}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    if not is_current:
                        _switch_thread(t["thread_id"])
                        st.rerun()
            with cols[1]:
                with st.popover("⋮", use_container_width=True):
                    n = t["n_messages"]
                    plural = "" if n == 1 else "s"
                    st.caption(f"{n} message{plural}")
                    st.caption(_format_when(t["updated_at"]))
                    st.divider()
                    if st.button(
                        "Xoá hội thoại",
                        key=f"del-{t['thread_id']}",
                        use_container_width=True,
                    ):
                        store_delete_thread(settings.chats_db_path, t["thread_id"])
                        if is_current:
                            _start_new_thread()
                        st.rerun()
    else:
        st.caption("Chưa có hội thoại nào được lưu.")

    st.divider()

    st.markdown("<div class='vt-section-label'>Câu hỏi mẫu</div>", unsafe_allow_html=True)
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
        # 1) Persist + render user message immediately so it survives a reload.
        save_turn(
            settings.chats_db_path,
            thread_id=st.session_state.thread_id,
            role="user",
            content=prompt,
        )
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

        # 3) Persist the assistant turn and append to session history.
        save_turn(
            settings.chats_db_path,
            thread_id=st.session_state.thread_id,
            role="assistant",
            content=final_answer,
            trace=final_trace,
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": final_answer, "trace": final_trace}
        )
        # Rerun so the sidebar "Past conversations" list refreshes immediately.
        st.rerun()
