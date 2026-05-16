# Test Plan & Results Summary

## Strategy

The test suite is split into three layers, all runnable offline (no Azure key required):

| Layer | What it verifies | Substitutes used |
| --- | --- | --- |
| **Unit** | Markdown ingestion, paragraph chunking, FAISS retriever round-trip, each of the 4 tools in isolation. | Real FAISS, deterministic `HashingEmbedder`, real seeded SQLite. |
| **Integration (agent loop)** | LangGraph wires LLM ↔ tools correctly: routing, multi-tool sequences, observation propagation, final answer extraction. | `FakeToolCallingChatModel` scripts deterministic tool calls. |
| **Manual / live (Streamlit)** | Real Azure LLM produces well-grounded, cited answers for the 8 scenarios in `sample_queries.json`. | None — run after `streamlit run app.py`. |

Hashing embedder is fine for unit tests because the test queries deliberately contain rare overlap words from the target chunks (e.g. "halong cruise overnight kayak" → unique to the Halong document).

## Automated test cases

All 29 tests run with `pytest tests/` in under 1 second. See `tests/` for source.

| Test ID | File | Verifies | Status |
| --- | --- | --- | --- |
| TC_ING_01 | test_ingestion.py | YAML scalar + list front-matter parsing | ✅ Pass |
| TC_ING_02 | test_ingestion.py | Multiple docs in one file are split correctly | ✅ Pass |
| TC_ING_03 | test_ingestion.py | Paragraph packing respects `max_chars` | ✅ Pass |
| TC_ING_04 | test_ingestion.py | Oversized single paragraph stays as one chunk | ✅ Pass |
| TC_ING_05 | test_ingestion.py | Real KB directory produces > 20 chunks with expected titles | ✅ Pass |
| TC_ING_06 | test_ingestion.py | Missing directory raises `FileNotFoundError` | ✅ Pass |
| TC_RET_01 | test_retriever.py | Top-1 hit is the semantically closest chunk | ✅ Pass |
| TC_RET_02 | test_retriever.py | `filter_fn` restricts results to matching metadata | ✅ Pass |
| TC_RET_03 | test_retriever.py | Empty / whitespace query returns `[]` | ✅ Pass |
| TC_RET_04 | test_retriever.py | `save()` then `load()` round-trip reproduces hits | ✅ Pass |
| TC_RET_05 | test_retriever.py | `load()` with missing files raises `FileNotFoundError` | ✅ Pass |
| TC_RET_06 | test_retriever.py | `list_titles()` deduplicates and preserves order | ✅ Pass |
| TC_TOOL_01 | test_tools.py | `search_travel_knowledge` returns ≥ 1 Halong-cruise hit | ✅ Pass |
| TC_TOOL_02 | test_tools.py | `region` filter narrows to the requested region | ✅ Pass |
| TC_TOOL_03 | test_tools.py | Empty-query RAG call returns `{count: 0, results: []}` | ✅ Pass |
| TC_TOOL_04 | test_tools.py | SQL tool: cruise + north filter on `tours` | ✅ Pass |
| TC_TOOL_05 | test_tools.py | SQL tool: Hoi An hotels under 2M VND | ✅ Pass |
| TC_TOOL_06 | test_tools.py | SQL tool: HAN → PQC flights | ✅ Pass |
| TC_TOOL_07 | test_tools.py | SQL tool rejects unknown tables | ✅ Pass |
| TC_TOOL_08 | test_tools.py | `estimate_trip_budget`: arithmetic + contingency match expected totals | ✅ Pass |
| TC_TOOL_09 | test_tools.py | `estimate_trip_budget` rejects negative nights | ✅ Pass |
| TC_TOOL_10 | test_tools.py | `web_search_news` returns `mode: "offline"` when no TAVILY_API_KEY | ✅ Pass |
| TC_AGT_01 | test_agent.py | Agent routes to `search_travel_knowledge`; observation captured | ✅ Pass |
| TC_AGT_02 | test_agent.py | Agent routes to `query_tour_inventory` for flight question | ✅ Pass |
| TC_AGT_03 | test_agent.py | Agent calls `estimate_trip_budget` and surfaces total in trace | ✅ Pass |
| TC_AGT_04 | test_agent.py | Empty user question yields blank answer (no tool call) | ✅ Pass |
| TC_AGT_05 | test_agent.py | Multi-tool turn captures both calls in order | ✅ Pass |
| TC_STR_01 | test_agent.py | `run_turn_stream` emits tool_call → tool_result → answer_chunk → final events | ✅ Pass |
| TC_STR_02 | test_agent.py | Blank question short-circuits to a single `final` event | ✅ Pass |

## Manual / live test scenarios

These exercise the actual Azure GPT-4o-mini model on the 8 prompts in `sample_queries.json`. Run after `streamlit run app.py`.

| Scenario | Question | Expected tool(s) | Acceptance |
| --- | --- | --- | --- |
| **S01** | "Mình có 3 ngày ở Hà Nội thì nên đi đâu, ăn gì?" | `search_travel_knowledge` | Reply in Vietnamese, ≥ 3 specific places + ≥ 3 food picks, each cited with `(doc_id)`. |
| **S02** | "I'm Australian, do I need a visa for a 2-week trip to Vietnam in March 2026?" | `search_travel_knowledge` | English reply, mentions Australia not on exemption list + recommends e-visa. Cites visa docs. |
| **S03** | "Find me the cheapest economy flight from Hanoi to Phu Quoc." | `query_tour_inventory` | Returns the lowest-price row (Vietjet VJ451 at 1,650,000 VND given current seed). |
| **S04** | "Tìm khách sạn 4 sao có ăn sáng ở Đà Nẵng dưới 2 triệu/đêm." | `query_tour_inventory` | Returns hotel `H011` (Brilliant Hotel Da Nang, 1,450,000 VND). |
| **S05** | Hotel + cruise + flight budget for 2 people, 6 nights. | `estimate_trip_budget` | Reply includes total VND + USD + per-person breakdown. |
| **S06** | "Plan a 7-day Vietnam itinerary for 2 people, budget USD 1,500 total. Show me a sample plan and the budget calculation." | `search_travel_knowledge` + `estimate_trip_budget` | Itinerary cited from KB + budget tool output presented as table or bullets. |
| **S07** | "Are there any current weather advisories for Halong Bay this week?" | `web_search_news` | Tool returns `mode: "offline"`; reply tells user live data not configured in demo. |
| **S08** | "What's the stock price of Apple today?" | (none) | Polite redirect to Vietnam travel topics, no fabricated stock number. |

## Robustness / edge-case checks

| Risk | Mitigation in code | Verified by |
| --- | --- | --- |
| Empty / whitespace user query | `run_turn` short-circuits before LLM call | TC_AGT_04 |
| LLM stuck in tool-call loop | `recursion_limit=12` cap on the LangGraph executor | Code review (settings.py + agent.py) |
| FAISS index not built | `Retriever.load` raises `FileNotFoundError` with build-script hint | TC_RET_05 |
| SQLite DB missing | Tool raises with seed-script hint | Code review (tools.py:_db_connect) |
| SQL injection via filter values | Whitelist of tables + parameterised queries with `?` placeholders | Code review (tools.py:query_tour_inventory) |
| Embedding API rate-limit / timeout | `tenacity` retry with exponential backoff (5 attempts) | Code review (embeddings.py:_embed_batch) |
| Tavily key missing | Tool returns structured `mode: "offline"` notice | TC_TOOL_10 |

## How to re-run

```bash
cd 03-Implementation/hackathon
source .venv/bin/activate
pytest tests/ -v
```

Expected output: `29 passed in <1s`.

## Smoke test

For graders running the project for the first time:

```bash
make smoke              # or:  bash scripts/smoke_test.sh
```

This re-verifies:

1. All modules import.
2. SQLite booking DB seeds with the expected 12/17/13 row counts.
3. All 29 pytest cases pass.
4. `app.py` parses without syntax errors.

Runs in ~20 seconds, no Azure key required.
