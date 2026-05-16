# Interface Screenshots

This directory holds the 5 required screenshots for the hackathon "Interface Screenshots" deliverable.

## Capture options

| Option | When to use | Command |
| --- | --- | --- |
| **Automated** | If you have a valid `OPENAI_API_KEY` and want reproducible captures. | `make screenshots` (installs Playwright + Chromium, opens the app headlessly, walks through 5 scenarios, saves PNGs here). |
| **Manual** | Quickest path for graders / demo day. | `streamlit run app.py`, open `http://localhost:8501`, capture per the checklist below with your OS screenshot tool. |

## Required screenshots

| File | What it should show |
| --- | --- |
| `01-initial-state.png` | Empty chat with the sidebar visible (API key field, **✓ Agent ready** badge, conversation list, sample-query expanders). |
| `02-rag-vietnamese.png` | Reply to S01 ("Mình có 3 ngày ở Hà Nội...") — assistant message in Vietnamese with `(doc_id)` citations. |
| `03-tool-call-expanded.png` | Same turn as #02 but with the **🔧 N tool call(s)** expander opened so the `search_travel_knowledge` args + raw observation are visible. |
| `04-multi-tool-budget.png` | Reply to S05 or S06 — assistant message containing both an itinerary/snippets AND a budget breakdown, with the tool-trace expander showing 2+ tool calls. |
| `05-out-of-domain-redirect.png` | Reply to S08 ("Apple stock price?") — assistant politely redirecting to Vietnam travel. |

Keep each screenshot ≤ 1.5 MB; PNG is fine.

## Embedding in deliverables

Once captured, the screenshots will be referenced from:

- [README.md](../../README.md) — quick-look thumbnails.
- [docs/02-mvp-features.md](../02-mvp-features.md) — illustrating each feature.
- [slides/presentation.md](../../slides/presentation.md) — demo flow slide.

Use the existing markdown links pattern: `![alt](docs/screenshots/02-rag-vietnamese.png)`.
