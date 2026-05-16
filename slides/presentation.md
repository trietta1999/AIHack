<!--
Marp-friendly slide deck.
Export to PDF / PPTX with:
   npx @marp-team/marp-cli@latest slides/presentation.md --pdf
or open in VS Code with the Marp extension.
-->

---
marp: true
title: Vietnam Travel Planner — RAG + LangGraph
paginate: true
---

# 🇻🇳 Vietnam Travel Planner

A domain-specific AI assistant for planning trips to Vietnam — built on **RAG + LangGraph + FAISS** with four tools.

*Hackathon: Building an Intelligent Domain-Specific AI Assistant with RAG System*
Group 2 — AIA_05

---

## Why "Vietnam Travel Planner"?

- **High-value domain**: visitors need destinations, transport, visa, budget, and live availability in one conversation.
- **Rich, finite knowledge surface**: fits into a curated KB of 31 chunks (7 markdown files) without dragging in millions of irrelevant docs.
- **Exercises every category in the brief**: RAG, SQL-style lookup, calculation, optional web search — all in one demo.
- **Bilingual VN/EN**: showcases the assistant's ability to track the user's language across multi-turn chats.

---

## Domain requirements

| Concern | What users ask | Why a generic LLM falls short |
| --- | --- | --- |
| Destinations | "What's worth doing in 3 days in Hanoi?" | Generic answers omit specific must-sees, prices, and seasonal warnings. |
| Visa & entry | "Do Australians need a visa for 14 days?" | Frontier policies (e.g. Phu Quoc 30-day exemption) change yearly. |
| Booking | "Cheapest flight HAN → PQC tomorrow" | LLM has no live inventory. |
| Budget | "Total for 6 nights + flight + tour?" | LLM does arithmetic poorly and inconsistently. |
| Real-time | "Typhoon advisory for Halong this week?" | LLM training data is stale. |

---

## Tech stack

- **LLM**: OpenAI `gpt-4o-mini` (chat) + `text-embedding-3-small` (RAG embeddings).
- **Agent**: LangGraph `create_react_agent` with `MemorySaver` checkpointer (multi-turn per `thread_id`), token-level streaming via `run_turn_stream`.
- **Vector store**: **FAISS** (`IndexFlatIP`, L2-normalised → cosine).
- **Knowledge base**: 7 markdown files (destinations, transport, visa, food, safety, accommodation, itineraries) with YAML front-matter — 26 documents → 31 chunks.
- **Booking inventory**: SQLite seeded with 12 tours, 17 hotels, 13 flights.
- **Frontend**: Streamlit — chat UI + sidebar (key, sample queries, past threads) + tool-trace expanders, with **persistent multi-thread history** (`data/chats.sqlite`, URL-shareable via `?t=<uuid>`).
- **Testing**: pytest with `HashingEmbedder` + `FakeToolCallingChatModel` (29 tests, run in < 1s, no API key needed).

---

## RAG pipeline

```text
data/knowledge_base/*.md  (7 files)
        |
        v
ingestion.chunks_from_dir (max 1200 chars/chunk)
        |
        v
OpenAI embeddings text-embedding-3-small (batched, retry)
        |
        v
L2-normalise → FAISS IndexFlatIP → data/faiss_index/
        |
        v
search(query, top_k, filter_fn) → cosine top-k chunks + scores
```

Citations: every retrieved chunk carries its `doc_id`, the system prompt forces `(doc_id)` mentions in the answer.

---

## The four tools

| Tool | Backing | What the LLM uses it for |
| --- | --- | --- |
| `search_travel_knowledge` | FAISS retriever | Open-ended travel Q&A (destinations, visa, food, safety…). |
| `query_tour_inventory` | SQLite (whitelist + parameter binding) | Real-time inventory: cheapest flight, hotels by city + star rating, tours by region + category. |
| `estimate_trip_budget` | Pure Python | Deterministic per-line + total budget in VND and USD. |
| `web_search_news` | Tavily (optional) | Live news / weather advisories. Gracefully returns `mode: "offline"` when no key. |

System prompt enforces: cite every fact from RAG, never invent prices/policies, use the calculator instead of computing in head, stay in the user's language.

---

## Test results

- **29 / 29 automated tests pass** in < 1s — `pytest tests/`.
  - 6 ingestion tests (YAML, paragraph packing, KB shape).
  - 6 retriever tests (FAISS round-trip, filters, persistence).
  - 10 tool tests (every tool, including SQL whitelist enforcement and offline web-search behaviour).
  - 7 agent tests (routing, multi-tool sequences, observation propagation, **streaming events**).
- **8 manual scenarios** in `sample_queries.json`, including a Vietnamese-only flow, an out-of-domain guardrail, and a budget-calc combo.
- Mocks: `HashingEmbedder` (deterministic MD5-of-tokens) + `FakeToolCallingChatModel` (scripted AIMessages) drive the full pipeline without an OpenAI key.
- Smoke test (`make smoke`) verifies imports + DB seed + tests + Streamlit syntax in ~20s.

---

## Deployment

Localhost only (per brief):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # paste OPENAI_API_KEY
python scripts/build_index.py      # build FAISS from KB
python scripts/seed_db.py          # seed bookings.sqlite
streamlit run app.py
```

Cloud-ready: same package runs on Streamlit Community Cloud, HF Spaces, or any container host via the shipped Docker image (see [docs/05-deployment.md](../docs/05-deployment.md)).

---

## Live demo (5 sample queries)

1. **S01** — VN: "Mình có 3 ngày ở Hà Nội thì nên đi đâu, ăn gì?" → RAG with citations.
2. **S03** — EN: "Cheapest flight Hanoi → Phu Quoc?" → SQL hit.
3. **S05** — EN: budget question with concrete prices → calculator.
4. **S06** — EN: "Plan a 7-day Vietnam itinerary, USD 1,500 budget" → RAG + calculator chained.
5. **S08** — Out-of-domain ("Apple stock price?") → polite redirect.

Each turn: open the **🔧 N tool call(s)** expander to show the raw observation.

---

## Evaluation against the rubric

| Criterion | Points | Self-assessment |
| --- | --- | --- |
| Innovation & Use Case | 20 | Original domain (Vietnam travel), bilingual, multi-tool agent, streaming responses, persistent multi-thread chat history. |
| RAG Pipeline | 25 | YAML-frontmatter KB, paragraph chunking, FAISS IndexFlatIP, L2-normalised cosine, persistent index, citations enforced. |
| User Interface Quality | 15 | Streamlit chat + sidebar sample queries + tool-trace expander + persistent thread list + URL-shareable thread ids. |
| LLM Response Relevance | 15 | System-prompt rules: cite, don't fabricate, stay in user's language; budget tool prevents arithmetic errors. |
| Testing & Robustness | 10 | 29 offline tests; explicit guard rails for missing index, missing DB, empty query, unknown SQL table, missing Tavily key. |
| Deployment & Demo | 15 | One-command Streamlit run; clear demo script in `docs/05-deployment.md`. |

---

## What's next (post-hackathon ideas)

- **Persistent vector store** (Pinecone or pgvector) for multi-user shared deployments.
- **Live inventory APIs**: Skyscanner / Booking / Agoda partner feeds instead of SQLite seed.
- **Voice mode**: HuggingFace TTS/STT for hands-free use.
- **Cross-country expansion**: Cambodia / Laos / Thailand KBs — architecture supports new markdown packs without code changes.
- **User profiles**: save trips, share itinerary links, export to PDF.

---

# Thanks

Source: hackathon GitLab repo, branch `Group2`.
Docs: [01-user-stories](../docs/01-user-stories.md) · [02-mvp-features](../docs/02-mvp-features.md) · [03-architecture](../docs/03-architecture.md) · [04-test-plan](../docs/04-test-plan.md) · [05-deployment](../docs/05-deployment.md)
