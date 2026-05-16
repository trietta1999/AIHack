# Hackathon — Vietnam Travel Planner (RAG + LangGraph + FAISS)

> Hackathon: *Building an Intelligent Domain-Specific AI Assistant with RAG System.* Group 2, AIA_05.

A bilingual (Vietnamese / English) travel-planning chatbot for Vietnam. Built on **OpenAI GPT-4o-mini + text-embedding-3-small**, **FAISS** vector store, **LangGraph** ReAct agent with **four tools** (RAG retrieval, SQLite booking lookup, trip-budget calculator, optional Tavily web search), and a **Streamlit** chat UI with **multi-thread chat persistence** (SQLite).

## At a glance

| Aspect | Detail |
| --- | --- |
| Domain | Vietnam travel — destinations, transport, visa, food, safety, accommodation, sample itineraries. |
| Knowledge base | 7 markdown files (`data/knowledge_base/`) → 26 documents → **31 FAISS chunks** (1,536-dim). |
| Inventory | SQLite seeded with **12 tours + 17 hotels + 13 flights**. |
| Tools (4) | `search_travel_knowledge` (RAG), `query_tour_inventory` (SQL whitelist), `estimate_trip_budget` (calculator), `web_search_news` (Tavily, optional). |
| Frontend | Streamlit chat with **streaming responses**, live tool-call status, sidebar sample queries, per-turn tool-trace expander, **persistent multi-thread history**, URL-shareable threads (`?t=<uuid>`). |
| Tests | **29 offline tests** (pytest, < 1s) — no OpenAI key required. |
| Deployment | Localhost (`make demo`), **Dockerfile** + **docker-compose.yml**, smoke-test script, screenshot capture script. |
| Deliverables | [User stories](docs/01-user-stories.md) · [MVP features](docs/02-mvp-features.md) · [Architecture diagrams](docs/03-architecture.md) · [Test plan](docs/04-test-plan.md) · [Deployment & demo](docs/05-deployment.md) · [Slides](slides/presentation.md). |

## Project layout

```text
03-Implementation/hackathon/
├── README.md                          # This file
├── Makefile                           # make install/seed/index/test/demo/docker/...
├── Dockerfile                         # Single-stage Python 3.12 image
├── docker-compose.yml                 # One-command stack with env passthrough
├── requirements.txt
├── .env.example
├── .gitignore .dockerignore
├── app.py                             # Streamlit entry-point (streaming + tool trace)
├── sample_queries.json                # 8 demo scenarios
├── data/
│   ├── knowledge_base/                # 7 markdown KB files (destinations,
│   │                                  #   transport, visa, food, safety,
│   │                                  #   accommodation, itineraries)
│   ├── faiss_index/                   # Built by scripts/build_index.py (ignored)
│   ├── bookings.sqlite                # Built by scripts/seed_db.py (ignored)
│   └── chats.sqlite                   # Per-thread chat history (auto-created on first run)
├── scripts/
│   ├── build_index.py                 # KB → FAISS
│   ├── seed_db.py                     # Booking inventory seed
│   ├── smoke_test.sh                  # Fresh-checkout verification
│   ├── capture_screenshots.py         # Headless Playwright capture of 5 demo shots
│   └── export_slides.sh               # Marp CLI → presentation.pdf/.pptx
├── travel_advisor/
│   ├── __init__.py
│   ├── config.py                      # Env-overridable settings dataclass
│   ├── models.py                      # Chunk / RetrievedChunk
│   ├── ingestion.py                   # YAML-frontmatter parser + chunker
│   ├── embeddings.py                  # OpenAIEmbedder + HashingEmbedder (tests)
│   ├── retriever.py                   # FAISS IndexFlatIP wrapper + persistence
│   ├── tools.py                       # 4 LangChain @tool wrappers
│   ├── chat_store.py                  # SQLite-backed chat thread persistence
│   └── agent.py                       # LangGraph agent + run_turn + run_turn_stream
├── docs/
│   ├── 01-user-stories.md
│   ├── 02-mvp-features.md
│   ├── 03-architecture.md
│   ├── 04-test-plan.md
│   ├── 05-deployment.md
│   └── screenshots/                   # 5 PNGs (capture via `make screenshots`)
├── slides/
│   └── presentation.md                # Marp-friendly deck (export via `make slides`)
└── tests/                             # 29 offline tests
    ├── conftest.py                    # Hashing embedder + fake chat model
    ├── test_ingestion.py
    ├── test_retriever.py
    ├── test_tools.py
    └── test_agent.py                  # Includes streaming tests
```

## Requirements

- **Python 3.10 / 3.11 / 3.12** (NOT 3.13 / 3.14 — LangGraph 0.3 + Pydantic 2 not fully supported there yet).
- An **OpenAI API key** with access to both `gpt-4o-mini` (chat) and `text-embedding-3-small` (embedding) models.
- (Optional) `TAVILY_API_KEY` for the `web_search_news` tool. The app boots without it.

## Quick start

Three equivalent paths — pick whichever you prefer.

### Option A — Makefile (recommended)

```bash
cd 03-Implementation/hackathon
cp .env.example .env                   # paste OPENAI_API_KEY=...
make demo                              # install + seed + index (if needed) + run Streamlit
```

Other one-command targets: `make test`, `make smoke`, `make docker`, `make slides`, `make screenshots`, `make clean`.

### Option B — Plain Python

```bash
cd 03-Implementation/hackathon
python3.12 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # paste OPENAI_API_KEY=...
python scripts/build_index.py                            # one-time: KB → FAISS (~10s)
python scripts/seed_db.py                                # one-time: seed booking inventory
streamlit run app.py                                     # http://localhost:8501
```

### Option C — Docker

```bash
cd 03-Implementation/hackathon
export OPENAI_API_KEY=...
docker compose up --build                                # http://localhost:8501
```

In the sidebar, paste your OpenAI key — the agent auto-boots and shows a green **✓ Agent ready** badge. Then ask questions or click any **Câu hỏi mẫu** (sample query) button. Responses stream live, with a per-turn status badge while tools run, and every thread is auto-saved to `data/chats.sqlite` (use the sidebar to switch between past conversations).

## Run the tests

```bash
make test          # or:  pytest tests/ -v
```

Expected: `29 passed`. No OpenAI key required — the suite uses a deterministic `HashingEmbedder` for the retriever and a scripted `FakeToolCallingChatModel` for the agent + streaming loop.

## Smoke test for graders

```bash
make smoke
```

Verifies module imports, DB seeding, all 29 tests, and Streamlit app syntax in ~20 seconds.

## Configuration

Defaults are in [`travel_advisor/config.py`](travel_advisor/config.py). Override any value via environment variables before launching (the values below are the actual defaults — override only what you need):

```bash
OPENAI_API_KEY=...                                          # required
OPENAI_CHAT_MODEL=gpt-4o-mini                               # optional (default)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small               # optional (default)
CHUNK_MAX_CHARS=1200                                        # optional (default)
TOP_K_DEFAULT=4                                             # optional (default)
AGENT_RECURSION_LIMIT=12                                    # optional (default)
AGENT_TEMPERATURE=0.2                                       # optional (default)
TAVILY_API_KEY=tvly-...                                     # enables web_search_news
streamlit run app.py
```

## How it works (short version)

1. **Build phase** (`scripts/build_index.py`): markdown files → YAML-frontmatter docs → paragraph-packed chunks (max 1200 chars) → OpenAI embeddings (`text-embedding-3-small`) → L2-normalised float32 matrix → FAISS `IndexFlatIP` → persisted to `data/faiss_index/` with a JSON metadata sidecar.
2. **Query phase** (every chat turn): user question → LangGraph ReAct agent → OpenAI GPT-4o-mini decides which of the 4 tools to call (often more than one in one turn) → tool observations are appended to the message stream → LLM produces a final cited answer → Streamlit renders the answer with a tool-trace expander.
3. **Persistence**: FAISS index + bookings DB live on disk; chat threads (user/assistant turns plus tool traces) are persisted to `data/chats.sqlite` via `travel_advisor.chat_store`, with the active `thread_id` carried in the URL (`?t=<uuid>`) so reloads and shared links restore the same conversation. `MemorySaver` keeps the LangGraph in-memory state per thread for the duration of the Streamlit process.

The full data + control flow with sequence diagram is in [docs/03-architecture.md](docs/03-architecture.md).

## Highlights

- **Source-id citations**: every fact in the assistant's reply traces back to a `doc_id` like `(dest-hanoi)` or `(visa-evisa)` — easy to audit.
- **Zero SQL-injection surface**: the SQL-style tool only accepts a whitelist of tables and binds filters as parameterised `?` placeholders.
- **Calculator over guessing**: the LLM is forced to use `estimate_trip_budget` so the user sees the line-by-line totals rather than a hallucinated number.
- **Offline-friendly Tavily**: when no `TAVILY_API_KEY` is configured, the tool returns a structured `mode: "offline"` notice and the assistant tells the user live data isn't available — no fabrication.
- **Deterministic test harness**: `HashingEmbedder` + `FakeToolCallingChatModel` let CI verify the entire agent loop without an API key.
- **Bilingual by default**: the system prompt instructs the agent to reply in the user's language; the test scenarios cover both VN and EN flows.

## Deliverables index

| Deliverable from hackathon brief | Location |
| --- | --- |
| User Stories & Use Case Documentation | [docs/01-user-stories.md](docs/01-user-stories.md) |
| MVP Feature List | [docs/02-mvp-features.md](docs/02-mvp-features.md) |
| System Architecture Diagrams | [docs/03-architecture.md](docs/03-architecture.md) |
| Interface Screenshots | Capture from running `streamlit run app.py` (the demo script in [docs/05-deployment.md](docs/05-deployment.md) lists what to capture). |
| Test Plan and Results Summary | [docs/04-test-plan.md](docs/04-test-plan.md) |
| Source Code Repository | this directory; pushed to branch `Group2` of the hackathon GitLab repo. |
| (Optional) Deployed URL | localhost only per brief; cloud options in [docs/05-deployment.md](docs/05-deployment.md). |
| Presentation Slide Deck | [slides/presentation.md](slides/presentation.md) (Marp-compatible). |

## License

For internal AIA_05 coursework use.
