# User Stories & Use Cases — Vietnam Travel Planner

## Personas

| Persona | Profile | Primary need |
| --- | --- | --- |
| **Anna — first-time foreign visitor** | Australian, 32, English-speaking, 10-day window in March 2026, mid-range budget. | One-stop assistant for itinerary, visa, transport, budgeting. |
| **Bao — Vietnamese domestic traveller** | HCMC resident, 28, plans long weekend trips to the north each quarter. | Quick comparison of tour and flight prices in Vietnamese. |
| **Carlos — backpacker** | Spanish, 24, 4-week budget trip, sleeps in hostels, rides scooters. | Cheapest hostels + visa-extension advice + street-food safety. |
| **Diem — travel agent assistant** | Hanoi-based agency, 35, books group trips for corporate clients. | Real-time inventory lookup for hotels/flights/tours + budget summaries. |

## User stories

### Discovery (RAG-powered Q&A)

1. **As Anna**, I want to ask "what should I see in Hanoi in 3 days?" so I get a sourced answer I can trust, with each fact citing a knowledge-base document id.
2. **As Carlos**, I want to ask "is Vietnam visa exempt for Spaniards?" and get an authoritative answer covering exemption duration and extension rules.
3. **As Anna**, I want to ask follow-up questions ("OK and what about food allergies?") that keep the same conversation context.
4. **As Bao**, I want to ask in Vietnamese ("Đi Sapa nên đi tháng nào?") and receive a Vietnamese reply.

### Booking inventory (SQL-style structured query)

5. **As Diem**, I want to find the cheapest economy flight from Hanoi to Phu Quoc for a specific day so I can quote a client.
6. **As Carlos**, I want to filter hostels in HCMC under 300k/night and see availability.
7. **As Anna**, I want to compare overnight Halong cruises by price tier (mid-range vs luxury).

### Trip planning (calculator + multi-tool)

8. **As Anna**, I want to combine "what should I do in Vietnam for 7 days?" with "estimate the cost" in one ask, and get an itinerary + budget breakdown.
9. **As Diem**, I want the assistant to compute per-person totals when I provide hotel/flight/tour prices, so the math is auditable.

### Real-time + guardrails

10. **As Anna**, I want to know if there are current weather advisories before a Halong cruise; if live data isn't available, I want a clear "not available" answer instead of a fabrication.
11. **As any user**, if I ask something outside Vietnam travel ("Apple stock price?"), I want to be steered back to travel topics — not refused, not lied to.

## Acceptance criteria summary

| Story | Tool exercised | Acceptance criterion |
| --- | --- | --- |
| 1, 2, 4 | `search_travel_knowledge` | Cites at least one `doc_id`, snippet from KB. |
| 3 | (chain memory) | Next turn uses prior context without re-asking. |
| 5, 6, 7 | `query_tour_inventory` | Returns rows from the live SQLite DB matching filters. |
| 8, 9 | `estimate_trip_budget` (+ RAG) | Breakdown with hotel/food/transport/contingency lines. |
| 10 | `web_search_news` | Returns `mode: "offline"` notice when TAVILY_API_KEY missing. |
| 11 | (system prompt) | Polite redirect to travel topics. |

Test cases that exercise each criterion are listed in [04-test-plan.md](04-test-plan.md).
