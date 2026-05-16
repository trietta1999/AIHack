# Deployment & Demo Instructions

## Three deployment paths

| Path | Best for | Time to first chat |
| --- | --- | --- |
| **A. Makefile (`make demo`)** | Graders + demo day. One command, picks up the venv + builds index + boots Streamlit. | ~90s on a fresh checkout (after `pip install`). |
| **B. Plain Python** | Familiarity, troubleshooting. Same steps, more visibility. | ~90s. |
| **C. Docker / docker-compose** | Reproducible cross-machine runs, cloud handoff. Single image, single port. | ~3 min first build, ~30s subsequent. |

## Local deployment (the demo path)

### Prerequisites

- **Python 3.10 / 3.11 / 3.12** (NOT 3.13 / 3.14 — LangGraph 0.3 + Pydantic 2 still catching up).
- ~300 MB disk space for the virtual environment.
- An **OpenAI API key** with access to both `gpt-4o-mini` (chat) and `text-embedding-3-small` (embedding) models.
- (Optional) A [Tavily](https://tavily.com) API key to enable the live `web_search_news` tool. The demo runs fine without it.
- (Optional, for Docker path) Docker 24+ and Docker Compose v2.

### Path A — Makefile (recommended)

```bash
cp .env.example .env                    # paste OPENAI_API_KEY=...
make demo
```

`make demo` creates the venv (if missing), installs requirements, seeds the SQLite DB, builds the FAISS index if absent, then launches `streamlit run app.py`.

Other useful targets:

- `make test` — run the 29 pytest cases (offline, no key).
- `make smoke` — fresh-checkout verification in ~20s.
- `make docker` — build + run the container.
- `make slides` — export `slides/presentation.{pdf,pptx}` via Marp.
- `make screenshots` — headless Playwright capture of the 5 demo screens.
- `make clean` — remove FAISS index, SQLite DB, pytest cache.

### Path B — Plain Python (manual)

```bash
# 1. Virtual env
python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Credentials
cp .env.example .env
$EDITOR .env                            # paste OPENAI_API_KEY=...

# 3. Build the FAISS index from the markdown KB
python scripts/build_index.py
#   → Saved FAISS index to data/faiss_index (dim=1536, n=31, ...)

# 4. Seed the SQLite booking DB (12 tours, 17 hotels, 13 flights)
python scripts/seed_db.py
#   → Seeded ... into data/bookings.sqlite

# 5. Run the app
streamlit run app.py
```

Open <http://localhost:8501>. Paste the same OpenAI key into the sidebar — the agent **auto-boots** and shows a green **✓ Agent ready** badge. Then use the **Câu hỏi mẫu** (sample query) buttons in the sidebar or type a question. Responses **stream live**, with a status badge while tools execute, and every conversation is auto-saved (use the sidebar list to switch between past threads or start a new one).

### Path C — Docker / docker-compose

```bash
export OPENAI_API_KEY=...              # required for the index build inside the container
# Optional: export TAVILY_API_KEY=tvly-...

docker compose up --build              # one command — http://localhost:8501
```

The container:

1. Seeds `data/bookings.sqlite` on first boot if missing.
2. Builds `data/faiss_index/` from the KB on first boot if missing (uses your OpenAI key).
3. Launches Streamlit on `0.0.0.0:8501` with a healthcheck on `/_stcore/health`.

`docker-compose.yml` mounts `./data` as a volume, so the FAISS index, bookings DB, and chats DB all persist across container restarts.

Plain Docker without compose:

```bash
docker build -t vn-travel-planner .
docker run --rm -p 8501:8501 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v "$(pwd)/data:/app/data" \
  vn-travel-planner
```

### Run the test suite

```bash
make test                              # or:  pytest tests/ -v
```

All 29 tests should pass in under a second; no API key is needed.

### Smoke test (graders)

```bash
make smoke
```

Verifies imports + DB seed + all tests + Streamlit syntax in ~20 seconds.

---

## Cloud deployment options (post-hackathon)

| Option | Recommendation | Notes |
| --- | --- | --- |
| **Streamlit Community Cloud** | Easiest. | Free tier; commit FAISS index + SQLite to the repo and configure secrets in the dashboard. |
| **Hugging Face Spaces (Streamlit SDK)** | Easy. | Same approach as Streamlit Cloud; supports private spaces. |
| **Render / Fly.io / Railway** | Production-ish. | Run the Docker image from this repo; set the OpenAI key as a secret env var. |
| **Docker** | Most portable. | A 4-line Dockerfile (`python:3.12-slim` → `pip install -r requirements.txt` → `streamlit run app.py`) is enough. Mount `data/` as a volume if you want the index/DB to persist across container restarts. |

A working [Dockerfile](../Dockerfile) and [docker-compose.yml](../docker-compose.yml) are shipped with the project (see Path C above). For cloud rollouts:

- **Streamlit Community Cloud / HF Spaces** — point the platform at `app.py`; add `OPENAI_API_KEY` (and optionally `TAVILY_API_KEY`) as platform secrets. Commit the FAISS index + SQLite for fastest cold-start, or build them in a post-deploy hook.
- **Container hosts (Render / Fly.io / Railway / etc.)** — `docker push` the image to the registry of your choice, point the service at it, set the same env vars.
- **Persistent volume**: the container expects `/app/data` to be writable. On Streamlit Cloud you'll need to either commit the prebuilt index or run `scripts/build_index.py` at startup (the entrypoint already handles this).

---

## Slides & screenshots

The hackathon deliverables include a slide deck and interface screenshots. Both are scriptable:

```bash
make slides         # → slides/presentation.pdf + .pptx (needs Node.js)
make screenshots    # → docs/screenshots/01..05*.png (needs OpenAI key + Chromium)
```

If you prefer manual screenshots, the capture checklist is in [docs/screenshots/README.md](screenshots/README.md). The 5 required shots:

1. `01-initial-state.png` — empty chat + sidebar with sample queries.
2. `02-rag-vietnamese.png` — Vietnamese RAG answer with citations.
3. `03-tool-call-expanded.png` — Tool-call expander open, showing args + raw observation.
4. `04-multi-tool-budget.png` — multi-tool turn (itinerary + budget breakdown).
5. `05-out-of-domain-redirect.png` — Apple stock question redirected back to travel.

---

## Demo script (5-minute pitch)

**Opening (30s)** — "Vietnam Travel Planner is a domain-specific RAG assistant. It answers travel questions in Vietnamese and English, cites its sources, and chains four tools — RAG retrieval, SQL booking lookup, budget calculator, and live web search — so users can plan an entire trip in one chat."

**Demo flow** — work through these sample queries from the sidebar:

1. **S01 — RAG (Vietnamese)**: "Mình có 3 ngày ở Hà Nội thì nên đi đâu, ăn gì?" → expand the *🔧 N tool call(s)* panel to show the `search_travel_knowledge` invocation and the actual snippets retrieved.
2. **S03 — SQL**: "Cheapest flight Hanoi → Phu Quoc?" → expand to show the live SQLite hit with `sql` and `filters` echoed back.
3. **S05 — Calculator**: budget question with concrete numbers → expand to show the `estimate_trip_budget` breakdown.
4. **S06 — Multi-tool**: "Plan a 7-day Vietnam itinerary for 2 people, budget USD 1,500" → expand to show RAG + budget being chained in one turn.
5. **S08 — Out-of-domain guardrail**: "What's the stock price of Apple today?" → assistant politely redirects.

**Closing (30s)** — point to the *Architecture* slide, mention that all 29 tests pass offline (no API key), and call out the optional Tavily slot for live news.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Missing FAISS index files` | Index never built or `data/faiss_index/` deleted. | Re-run `python scripts/build_index.py`. |
| `Bookings DB missing` | DB never seeded or `data/bookings.sqlite` deleted. | `python scripts/seed_db.py`. |
| `OPENAI_API_KEY missing` | `.env` empty or sidebar field blank. | Edit `.env` or paste in sidebar. |
| `401 / 403` from OpenAI | Key invalid, or missing access to either model. | Confirm both `gpt-4o-mini` and `text-embedding-3-small` are accessible from your key. |
| Streamlit page blank | Streamlit cached a broken session state. | Click **＋ Hội thoại mới** in the sidebar, or restart `streamlit run app.py`. |
| `recursion_limit reached` | Agent stuck in a tool loop. | Reset conversation; consider raising `AGENT_RECURSION_LIMIT` env var if your real workflows need > 12 steps. |
| Tests fail on a fresh checkout | Forgot to seed the DB; conftest auto-seeds but only when import succeeds. | Run `python scripts/seed_db.py` then `pytest tests/`. |
