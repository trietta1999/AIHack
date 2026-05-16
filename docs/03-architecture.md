# System Architecture

## 1. High-level component diagram

```mermaid
flowchart LR
    user([User])
    ui[Streamlit Chat UI<br/>app.py]
    agent[LangGraph ReAct Agent<br/>travel_advisor.agent]
    llm[(OpenAI<br/>gpt-4o-mini)]
    emb[(OpenAI<br/>text-embedding-3-small)]

    subgraph Tools
        t1[search_travel_knowledge<br/>FAISS + cosine]
        t2[query_tour_inventory<br/>SQLite whitelist]
        t3[estimate_trip_budget<br/>pure Python calc]
        t4[web_search_news<br/>Tavily optional]
    end

    faiss[(FAISS Index<br/>data/faiss_index)]
    kb[(Knowledge Base<br/>data/knowledge_base/*.md)]
    db[(SQLite bookings<br/>data/bookings.sqlite)]
    chats[(SQLite chats<br/>data/chats.sqlite)]
    web((Public web<br/>via Tavily))

    user -->|chat| ui
    ui -->|HumanMessage| agent
    ui <-->|save/load turns| chats
    agent <-->|messages| llm
    agent -->|invoke| t1
    agent -->|invoke| t2
    agent -->|invoke| t3
    agent -->|invoke| t4
    t1 --> faiss
    t1 -->|query embed| emb
    faiss -.->|built once<br/>scripts/build_index.py| kb
    t2 --> db
    db -.->|seeded once<br/>scripts/seed_db.py| db
    t4 -.->|optional| web
    agent -->|AIMessage + trace| ui
    ui -->|render| user
```

## 2. RAG ingestion pipeline (one-time)

```mermaid
flowchart LR
    md[data/knowledge_base/*.md<br/>7 files] -->|YAML frontmatter parse| ing[ingestion.chunks_from_dir]
    ing -->|paragraph packing<br/>max_chars=1200| chunks[Chunk objects<br/>31 total]
    chunks -->|embed| oaemb[OpenAI embeddings<br/>text-embedding-3-small]
    oaemb -->|L2-normalised float32| vecs[NumPy matrix N x 1536]
    vecs -->|IndexFlatIP.add| faissidx[FAISS index]
    chunks -->|sidecar JSON| meta[chunks.json]
    faissidx -->|write_index| disk[(data/faiss_index/index.faiss)]
    meta --> disk2[(data/faiss_index/chunks.json)]
```

Run with:

```bash
python scripts/build_index.py
```

## 3. RAG query pipeline (per turn)

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant CS as chat_store (SQLite)
    participant A as Agent (LangGraph)
    participant LLM as OpenAI GPT-4o-mini
    participant RAG as search_travel_knowledge
    participant E as OpenAI embeddings
    participant F as FAISS index

    U->>UI: Question
    UI->>CS: save_turn(role="user")
    UI->>A: HumanMessage + thread_id (via run_turn_stream)
    A->>LLM: messages + tool specs
    LLM-->>A: tool_call(search_travel_knowledge, args)
    A-->>UI: StreamEvent(tool_call)
    A->>RAG: invoke(query, top_k)
    RAG->>E: embed([query])
    E-->>RAG: 1 x 1536 vector
    RAG->>F: search(qvec, top_k)
    F-->>RAG: top_k chunks + scores
    RAG-->>A: JSON observation
    A-->>UI: StreamEvent(tool_result)
    A->>LLM: append ToolMessage
    LLM-->>A: tool_call(estimate_trip_budget, args)<br/>or streamed final AIMessage tokens
    Note over A,LLM: Loop until LLM emits a final AIMessage (no more tool_calls).
    LLM-->>A: final AIMessage tokens
    A-->>UI: StreamEvent(answer_chunk) (live render)
    A-->>UI: StreamEvent(final) — TurnResult(answer, trace)
    UI->>CS: save_turn(role="assistant", trace=[...])
    UI-->>U: rendered answer + tool-trace expander
```

## 4. Module dependency map

```mermaid
flowchart TD
    app[app.py - Streamlit]
    cli[scripts/build_index.py<br/>scripts/seed_db.py]
    agent[travel_advisor.agent]
    tools[travel_advisor.tools]
    retr[travel_advisor.retriever]
    ing[travel_advisor.ingestion]
    emb[travel_advisor.embeddings]
    chat[travel_advisor.chat_store]
    mod[travel_advisor.models]
    cfg[travel_advisor.config]

    app --> agent
    app --> retr
    app --> chat
    app --> cfg
    app --> emb
    cli --> ing
    cli --> retr
    cli --> emb
    cli --> cfg
    agent --> tools
    agent --> cfg
    tools --> retr
    tools --> cfg
    retr --> mod
    retr --> emb
    ing --> mod
    emb --> cfg
```

## 5. Why these choices

| Decision | Rationale |
| --- | --- |
| **FAISS (IndexFlatIP) over Chroma/Pinecone** | Smallest dependency surface. Persists to a single binary file + JSON sidecar. No external service to provision for a hackathon demo. Easy to swap later — the `Retriever` API is intentionally narrow. |
| **L2-normalised vectors + inner product** | Reduces cosine similarity to a single matrix multiply; FAISS doesn't have a dedicated cosine index but does have IP. |
| **YAML front-matter markdown KB** | Plain-text, version-control friendly, lets us pack multiple docs in one file for editorial readability. Tiny custom parser (no PyYAML) keeps deps light. |
| **Parameterised SQLite tool (not SQLDatabaseToolkit)** | Zero SQL-injection surface. LLM picks filters from a documented schema; we generate the SQL ourselves with `?` placeholders. Trade-off: less expressive than raw SQL, more secure and predictable. |
| **`estimate_trip_budget` as a dedicated tool** | Forces the LLM to compute totals deterministically (vs hallucinating arithmetic) and surfaces the breakdown to the user. |
| **LangGraph `create_react_agent` + `MemorySaver`** | Mature pattern from coursework (assignment-13/14); supports multi-tool routing and multi-turn memory with one helper. |
| **Token-level streaming via `run_turn_stream`** | LangGraph's multi-mode `.stream(stream_mode=["updates", "messages"])` surfaces both tool-call events (`updates`) and per-token answer chunks (`messages`), so the UI can render live tool status + streaming text without polling. |
| **SQLite chat persistence (`chat_store`)** | Threads + per-turn tool traces survive process restarts. Each thread id is propagated through the URL (`?t=<uuid>`) so reloads and shared links restore the same conversation. Lightweight (one file, no service). |
| **Streamlit (not React)** | Single-process Python deployment matches the hackathon's "localhost" deployment story. ~450 LOC for a working chat UI with citations, streaming, persistence, and thread sidebar. |
| **HashingEmbedder for tests** | Deterministic, no API key, < 1s test runtime. Enables CI-friendly tests. |

## 6. API specification (agent tool contracts)

Each tool exposes a JSON-schema contract to the LLM via its `@tool` docstring. See [tools.py](../travel_advisor/tools.py) for the source of truth; below is the abbreviated form.

### `search_travel_knowledge`

```json
{
  "query": "string (required, free text)",
  "top_k": "integer (default 4)",
  "region": "enum: north | central | south | nationwide | null"
}
```

Returns:

```json
{
  "count": "int",
  "results": [
    {"doc_id": "str", "title": "str", "source_file": "str",
     "score": "float", "snippet": "str", "tags": ["..."], "region": "str"}
  ]
}
```

### `query_tour_inventory`

```json
{
  "table": "enum: tours | hotels | flights (required)",
  "filters": "object (see schema in docstring)",
  "limit": "integer 1-25 (default 5)"
}
```

Returns: `{count, results, sql, filters}` or `{error, allowed}`.

### `estimate_trip_budget`

```json
{
  "nights_hotel": "int (required, >=0)",
  "hotel_price_per_night_vnd": "int (required, >=0)",
  "flights_total_vnd": "int (default 0)",
  "tours_total_vnd": "int (default 0)",
  "daily_food_vnd": "int (default 400000)",
  "daily_transport_vnd": "int (default 200000)",
  "travellers": "int (default 1, >=1)",
  "contingency_pct": "int (default 10)"
}
```

Returns: per-line subtotals, grand total, per-person breakdown, USD conversion at VND_PER_USD = 25,500.

### `web_search_news`

```json
{
  "query": "string (required)",
  "max_results": "integer 1-10 (default 3)"
}
```

Returns: `{mode: "live"|"offline", count, results: [{title, url, snippet, published_at?}], note?}`.
