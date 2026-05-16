# MVP Feature List

| # | Feature | Status | Backing component |
| --- | --- | --- | --- |
| F1 | Natural-language Q&A on Vietnam travel (Vietnamese + English) | ✅ Implemented | `search_travel_knowledge` tool + OpenAI GPT-4o-mini |
| F2 | RAG over markdown knowledge base (7 files / 26 documents / 31 chunks) | ✅ Implemented | `travel_advisor/ingestion.py` + `travel_advisor/retriever.py` (FAISS IndexFlatIP) |
| F3 | Source-id citations in answers | ✅ Implemented | System prompt enforces `(doc_id)` citations |
| F4 | Live booking inventory lookup (tours / hotels / flights) | ✅ Implemented | `query_tour_inventory` tool + seeded SQLite |
| F5 | Trip budget calculator (per-person VND/USD) | ✅ Implemented | `estimate_trip_budget` tool |
| F6 | Optional Tavily web-search tool for time-sensitive queries | ✅ Implemented (offline-graceful) | `web_search_news` tool |
| F7 | Multi-tool routing in one turn (e.g. RAG + budget) | ✅ Implemented | LangGraph `create_react_agent` |
| F8 | Multi-turn conversation memory per thread | ✅ Implemented | `MemorySaver` checkpointer + `thread_id` |
| F9 | Streamlit chat UI with citations + tool-trace expanders | ✅ Implemented | [app.py](../app.py) |
| F10 | **Streaming responses** with live tool-call status | ✅ Implemented | `run_turn_stream` in [agent.py](../travel_advisor/agent.py) + `st.empty()` placeholders in [app.py](../app.py) |
| F11 | Out-of-domain guardrail (politely steers back to travel) | ✅ Implemented | System prompt |
| F12 | Sample query buttons for demo | ✅ Implemented | Sidebar in [app.py](../app.py), data in [sample_queries.json](../sample_queries.json) |
| F13 | **Persistent multi-thread chat history** (SQLite-backed, URL-shareable threads) | ✅ Implemented | [travel_advisor/chat_store.py](../travel_advisor/chat_store.py) + `?t=<uuid>` query param in [app.py](../app.py) |
| F14 | **One-command Docker deployment** | ✅ Implemented | [Dockerfile](../Dockerfile) + [docker-compose.yml](../docker-compose.yml) |
| F15 | **Makefile + smoke-test** for graders | ✅ Implemented | [Makefile](../Makefile) + [scripts/smoke_test.sh](../scripts/smoke_test.sh) |
| F16 | Automated screenshot capture (5 demo screens) | ✅ Implemented | [scripts/capture_screenshots.py](../scripts/capture_screenshots.py) |
| F17 | Slide export pipeline (Markdown → PDF + PPTX) | ✅ Implemented | [scripts/export_slides.sh](../scripts/export_slides.sh) |
| F18 | Offline test suite (29 tests, no API key needed) | ✅ Implemented | `tests/` with `HashingEmbedder` + `FakeToolCallingChatModel` |

## Out of scope (stretch ideas, not built)

- Real-time hotel/flight inventory from live OTA APIs (Booking, Agoda, Skyscanner).
- Voice input / output (TTS / STT).
- Persistent user profiles, saved trips, sharing links.
- A second domain expansion (e.g. Cambodia / Laos / Thailand) — by design the architecture supports this via adding new markdown files + re-indexing.
- Pinecone / Chroma persistent vector store. The FAISS implementation persists to disk (`data/faiss_index/`) so this is mostly a hosting decision.
- LangChain `SQLDatabaseToolkit` raw-SQL agent. We chose a parameterised whitelist instead to keep injection surface zero.
