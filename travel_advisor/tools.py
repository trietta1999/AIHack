"""LangChain `@tool` wrappers exposed to the LangGraph agent.

Four tools (one per category from the hackathon brief):

1. `search_travel_knowledge` — RAG over the curated markdown KB.
2. `query_tour_inventory` — SQL-style lookup over the bookings SQLite DB
   (mirrors what `SQLDatabaseToolkit` would do, but constrained to a few
   safe parameter-driven queries so the LLM can't run free-form SQL).
3. `estimate_trip_budget` — pure-Python calculator (stand-in for
   PythonREPL). Sum-of-segments arithmetic with VND / USD conversion.
4. `web_search_news` — Tavily-style live web search; gracefully returns
   a synthetic message when no Tavily API key is configured so the demo
   stays runnable offline.

Each tool's docstring is the spec the LLM sees, so it doubles as inline
documentation.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from langchain_core.tools import tool

from .config import settings
from .retriever import get_retriever

VND_PER_USD = 25500  # 2026 indicative; refresh manually if needed.


# ---------------------------------------------------------------------------
# 1) RAG tool
# ---------------------------------------------------------------------------


def _format_chunk(rc, *, snippet_chars: int = 600) -> dict[str, Any]:
    text = rc.chunk.text
    if len(text) > snippet_chars:
        text = text[:snippet_chars].rstrip() + "..."
    return {
        "doc_id": rc.chunk.doc_id,
        "title": rc.chunk.title,
        "source_file": rc.chunk.source_file,
        "score": round(rc.score, 4),
        "snippet": text,
        "tags": rc.chunk.metadata.get("tags", []),
        "region": rc.chunk.metadata.get("region", "nationwide"),
    }


@tool
def search_travel_knowledge(
    query: str,
    top_k: int = 4,
    region: str | None = None,
) -> dict[str, Any]:
    """Search the curated Vietnam travel knowledge base (RAG over markdown docs).

    Use this for any question about destinations, transport, food, visa,
    safety, accommodation, or sample itineraries. The retriever embeds the
    query with text-embedding-3-small and returns the top-k closest chunks
    by cosine similarity (range -1 to 1; >0.4 is usually relevant).

    Args:
        query: Free-text question in any language. Example: "How long
            should I spend in Hoi An?" or "tư vấn vé tàu Hanoi Lao Cai".
        top_k: Maximum number of chunks to return. Default 4.
        region: Optional filter on the front-matter `region` value.
            Allowed: "north", "central", "south", "nationwide". Pass null
            to search everywhere.

    Returns:
        Dict with `count` and `results` — each result has `doc_id`,
        `title`, `source_file`, `score`, `snippet` (first ~600 chars),
        `tags`, `region`. Empty results when nothing relevant is found.
    """
    retriever = get_retriever()
    region_norm = region.strip().lower() if region else None

    def _matches(chunk) -> bool:
        if not region_norm:
            return True
        return str(chunk.metadata.get("region", "")).lower() == region_norm

    hits = retriever.search(
        query,
        top_k=max(1, top_k),
        filter_fn=_matches if region_norm else None,
    )
    return {"count": len(hits), "results": [_format_chunk(h) for h in hits]}


# ---------------------------------------------------------------------------
# 2) SQL tool — restricted to parameterised queries
# ---------------------------------------------------------------------------


_ALLOWED_TABLES = {"tours", "hotels", "flights"}


def _db_connect() -> sqlite3.Connection:
    if not settings.bookings_db_path.exists():
        raise RuntimeError(
            f"Bookings DB missing at {settings.bookings_db_path}. "
            "Run `python scripts/seed_db.py` first."
        )
    conn = sqlite3.connect(settings.bookings_db_path)
    conn.row_factory = sqlite3.Row
    return conn


@tool
def query_tour_inventory(
    table: str,
    filters: dict[str, Any] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Look up real-time booking inventory in the SQLite booking database.

    Returns rows as JSON-serialisable dicts. Safer than raw SQL: only the
    `tours`, `hotels`, and `flights` tables are queryable, and `filters`
    are bound as SQL parameters (no injection surface).

    Args:
        table: One of "tours", "hotels", "flights".
        filters: Optional equality / range filters. Examples:
            {"region": "north", "category": "cruise"} on `tours`;
            {"city": "Hoi An", "star_rating_min": 4} on `hotels`;
            {"origin": "HAN", "destination": "DAD", "price_max_vnd": 1000000}
            on `flights`.
            Supported per-table keys:
              tours: region, category, duration_days_max, price_max_vnd
              hotels: city, star_rating_min, breakfast_included,
                      price_max_vnd
              flights: origin, destination, airline, cabin_class,
                       price_max_vnd, duration_max_minutes
        limit: Max rows to return. Default 5, hard-capped at 25.

    Returns:
        Dict with `count` (rows returned) and `results` (list of row
        dicts). On invalid `table` returns
        `{"error": "table 'x' not allowed", "allowed": [...]}`.
    """
    if table not in _ALLOWED_TABLES:
        return {"error": f"table '{table}' not allowed", "allowed": sorted(_ALLOWED_TABLES)}
    limit = max(1, min(25, int(limit)))
    where_parts: list[str] = []
    params: list[Any] = []
    f = dict(filters or {})

    if table == "tours":
        if "region" in f:
            where_parts.append("region = ?")
            params.append(f["region"])
        if "category" in f:
            where_parts.append("category = ?")
            params.append(f["category"])
        if "duration_days_max" in f:
            where_parts.append("duration_days <= ?")
            params.append(int(f["duration_days_max"]))
        if "price_max_vnd" in f:
            where_parts.append("price_vnd <= ?")
            params.append(int(f["price_max_vnd"]))
        order = "ORDER BY price_vnd ASC"
    elif table == "hotels":
        if "city" in f:
            where_parts.append("city = ?")
            params.append(f["city"])
        if "star_rating_min" in f:
            where_parts.append("star_rating >= ?")
            params.append(int(f["star_rating_min"]))
        if "breakfast_included" in f:
            where_parts.append("breakfast_included = ?")
            params.append(1 if bool(f["breakfast_included"]) else 0)
        if "price_max_vnd" in f:
            where_parts.append("price_per_night_vnd <= ?")
            params.append(int(f["price_max_vnd"]))
        order = "ORDER BY star_rating DESC, price_per_night_vnd ASC"
    else:  # flights
        if "origin" in f:
            where_parts.append("origin = ?")
            params.append(str(f["origin"]).upper())
        if "destination" in f:
            where_parts.append("destination = ?")
            params.append(str(f["destination"]).upper())
        if "airline" in f:
            where_parts.append("airline = ?")
            params.append(f["airline"])
        if "cabin_class" in f:
            where_parts.append("cabin_class = ?")
            params.append(f["cabin_class"])
        if "price_max_vnd" in f:
            where_parts.append("price_vnd <= ?")
            params.append(int(f["price_max_vnd"]))
        if "duration_max_minutes" in f:
            where_parts.append("duration_minutes <= ?")
            params.append(int(f["duration_max_minutes"]))
        order = "ORDER BY price_vnd ASC"

    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    sql = f"SELECT * FROM {table} {where} {order} LIMIT ?"
    params.append(limit)

    with _db_connect() as conn:
        cursor = conn.execute(sql, params)
        rows = [dict(r) for r in cursor.fetchall()]
    return {"count": len(rows), "results": rows, "sql": sql.strip(), "filters": f}


# ---------------------------------------------------------------------------
# 3) Calculator tool
# ---------------------------------------------------------------------------


@tool
def estimate_trip_budget(
    nights_hotel: int,
    hotel_price_per_night_vnd: int,
    flights_total_vnd: int = 0,
    tours_total_vnd: int = 0,
    daily_food_vnd: int = 400000,
    daily_transport_vnd: int = 200000,
    travellers: int = 1,
    contingency_pct: int = 10,
) -> dict[str, Any]:
    """Compute a Vietnam trip budget from its line items (pure arithmetic).

    Treat this as a deterministic calculator: pass concrete prices from
    the user's plan and tool results, get back per-person and total
    figures in VND and USD. **Always** call this rather than computing
    totals in your own reply — that way the user can see the breakdown.

    Args:
        nights_hotel: Number of paid hotel nights.
        hotel_price_per_night_vnd: Per-room price; assumes one room shared
            by `travellers`.
        flights_total_vnd: Sum of all flight tickets for the trip (across
            all travellers).
        tours_total_vnd: Sum of all booked tour prices (across all
            travellers).
        daily_food_vnd: Per-person daily food spend estimate. Default
            400,000 VND (mid-range).
        daily_transport_vnd: Per-person daily local transport estimate
            (Grab, taxis, scooter rental). Default 200,000 VND.
        travellers: Number of people splitting hotel cost. Default 1.
        contingency_pct: Buffer added to the subtotal. Default 10.

    Returns:
        Dict with per-line subtotals, grand total, per-person total, and
        a USD conversion at the configured VND_PER_USD rate. All amounts
        are integers (VND) or rounded floats (USD).
    """
    if nights_hotel < 0 or travellers < 1:
        return {"error": "nights_hotel must be >=0 and travellers >=1"}
    hotel_total = max(0, int(nights_hotel)) * max(0, int(hotel_price_per_night_vnd))
    food_total = travellers * (nights_hotel + 1) * max(0, int(daily_food_vnd))
    transport_total = travellers * (nights_hotel + 1) * max(0, int(daily_transport_vnd))
    subtotal = hotel_total + flights_total_vnd + tours_total_vnd + food_total + transport_total
    contingency = int(subtotal * max(0, contingency_pct) / 100)
    total = subtotal + contingency
    breakdown = {
        "hotel_total_vnd": hotel_total,
        "flights_total_vnd": int(flights_total_vnd),
        "tours_total_vnd": int(tours_total_vnd),
        "food_total_vnd": food_total,
        "transport_total_vnd": transport_total,
        "subtotal_vnd": subtotal,
        "contingency_vnd": contingency,
        "total_vnd": total,
        "per_person_vnd": total // travellers,
        "total_usd": round(total / VND_PER_USD, 2),
        "per_person_usd": round(total / VND_PER_USD / travellers, 2),
        "assumptions": {
            "travellers": travellers,
            "trip_days": nights_hotel + 1,
            "vnd_per_usd": VND_PER_USD,
            "contingency_pct": contingency_pct,
        },
    }
    return breakdown


# ---------------------------------------------------------------------------
# 4) Web search tool (Tavily-style, optional)
# ---------------------------------------------------------------------------


@tool
def web_search_news(query: str, max_results: int = 3) -> dict[str, Any]:
    """Fetch up-to-date travel news / advisories from the public web (Tavily).

    Use this only for time-sensitive questions the curated knowledge base
    cannot answer — weather alerts, current safety advisories, recent
    policy changes, news. Avoid for general "best time to visit" /
    "things to do" questions (use `search_travel_knowledge` for those).

    Args:
        query: Free-text web query. Be specific: include place + date
            window. E.g. "Halong Bay typhoon advisory October 2026".
        max_results: Number of results to return. Default 3.

    Returns:
        Dict with `count` and `results` (each `{title, url, snippet,
        published_at?}`). If no TAVILY_API_KEY is configured the tool
        returns a clear `mode: "offline"` notice; callers should tell the
        user that no live data is available rather than fabricating it.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "mode": "offline",
            "count": 0,
            "results": [],
            "note": (
                "Live web search is not configured (set TAVILY_API_KEY to enable). "
                "Tell the user you cannot fetch real-time news in this demo."
            ),
        }
    # Lazy import — keeps Tavily optional.
    try:
        from tavily import TavilyClient  # type: ignore[import-untyped]
    except ImportError:
        return {
            "mode": "offline",
            "count": 0,
            "results": [],
            "note": "TAVILY_API_KEY set but tavily-python not installed; "
                    "`pip install tavily-python` to enable.",
        }
    client = TavilyClient(api_key=api_key)
    raw = client.search(query=query, max_results=max(1, min(10, int(max_results))))
    results = []
    for r in raw.get("results", []):
        results.append({
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": (r.get("content") or "")[:600],
            "published_at": r.get("published_date"),
        })
    return {"mode": "live", "count": len(results), "results": results}


ALL_TOOLS = [
    search_travel_knowledge,
    query_tour_inventory,
    estimate_trip_budget,
    web_search_news,
]
