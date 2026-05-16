"""Direct unit tests for each of the four agent tools."""

from __future__ import annotations

import json

import pytest

from travel_advisor.tools import (
    estimate_trip_budget,
    query_tour_inventory,
    search_travel_knowledge,
    web_search_news,
)


# ---------------------------------------------------------------------------
# search_travel_knowledge
# ---------------------------------------------------------------------------


def test_search_travel_knowledge_returns_hits(installed_retriever):
    out = search_travel_knowledge.invoke(
        {"query": "Halong Bay cruise overnight kayak", "top_k": 3}
    )
    assert out["count"] >= 1
    titles = [r["title"].lower() for r in out["results"]]
    assert any("ha long" in t or "halong" in t for t in titles)
    for r in out["results"]:
        assert "doc_id" in r and "snippet" in r and "score" in r


def test_search_travel_knowledge_region_filter(installed_retriever):
    out = search_travel_knowledge.invoke(
        {"query": "beach island", "top_k": 5, "region": "south"}
    )
    for r in out["results"]:
        assert r["region"] == "south"


def test_search_travel_knowledge_empty_query(installed_retriever):
    out = search_travel_knowledge.invoke({"query": "", "top_k": 3})
    assert out == {"count": 0, "results": []}


# ---------------------------------------------------------------------------
# query_tour_inventory
# ---------------------------------------------------------------------------


def test_query_tour_inventory_filters_tours_by_region_and_category():
    out = query_tour_inventory.invoke(
        {
            "table": "tours",
            "filters": {"region": "north", "category": "cruise"},
            "limit": 5,
        }
    )
    assert out["count"] >= 1
    for row in out["results"]:
        assert row["region"] == "north"
        assert row["category"] == "cruise"


def test_query_tour_inventory_hotels_in_hoian_under_price():
    out = query_tour_inventory.invoke(
        {
            "table": "hotels",
            "filters": {"city": "Hoi An", "price_max_vnd": 2_000_000},
            "limit": 10,
        }
    )
    assert out["count"] >= 1
    for row in out["results"]:
        assert row["city"] == "Hoi An"
        assert row["price_per_night_vnd"] <= 2_000_000


def test_query_tour_inventory_flights_origin_destination():
    out = query_tour_inventory.invoke(
        {
            "table": "flights",
            "filters": {"origin": "HAN", "destination": "PQC"},
            "limit": 5,
        }
    )
    assert out["count"] >= 1
    for row in out["results"]:
        assert row["origin"] == "HAN"
        assert row["destination"] == "PQC"


def test_query_tour_inventory_rejects_unknown_table():
    out = query_tour_inventory.invoke({"table": "secrets", "filters": {}, "limit": 5})
    assert "error" in out
    assert "allowed" in out


# ---------------------------------------------------------------------------
# estimate_trip_budget
# ---------------------------------------------------------------------------


def test_estimate_trip_budget_basic_math():
    out = estimate_trip_budget.invoke(
        {
            "nights_hotel": 6,
            "hotel_price_per_night_vnd": 1_500_000,
            "flights_total_vnd": 2_600_000,   # 2pax * 650k * 2 segments
            "tours_total_vnd": 10_400_000,    # 2pax * 5.2M cruise
            "daily_food_vnd": 400_000,
            "daily_transport_vnd": 200_000,
            "travellers": 2,
            "contingency_pct": 10,
        }
    )
    # hotel: 6 * 1.5M = 9M; food: 2pax * 7d * 400k = 5.6M;
    # transport: 2pax * 7d * 200k = 2.8M
    assert out["hotel_total_vnd"] == 9_000_000
    assert out["food_total_vnd"] == 5_600_000
    assert out["transport_total_vnd"] == 2_800_000
    assert out["subtotal_vnd"] == 9_000_000 + 2_600_000 + 10_400_000 + 5_600_000 + 2_800_000
    # contingency is 10% of subtotal, rounded down.
    expected_total = out["subtotal_vnd"] + out["contingency_vnd"]
    assert out["total_vnd"] == expected_total
    assert out["per_person_vnd"] == expected_total // 2
    assert out["total_usd"] > 0
    assert out["assumptions"]["travellers"] == 2


def test_estimate_trip_budget_rejects_invalid_inputs():
    out = estimate_trip_budget.invoke(
        {"nights_hotel": -1, "hotel_price_per_night_vnd": 1, "travellers": 1}
    )
    assert "error" in out


# ---------------------------------------------------------------------------
# web_search_news
# ---------------------------------------------------------------------------


def test_web_search_news_offline_when_no_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    out = web_search_news.invoke({"query": "halong typhoon", "max_results": 2})
    assert out["mode"] == "offline"
    assert out["count"] == 0
    assert "note" in out
