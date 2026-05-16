"""Seed an SQLite database with mock Vietnam tour-booking inventory.

Used by the `query_tour_inventory` tool. Re-run any time to reset the DB.

    python scripts/seed_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_PATH = BASE / "data" / "bookings.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS tours (
    tour_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,        -- 'north' | 'central' | 'south'
    category TEXT NOT NULL,      -- 'cruise' | 'trek' | 'city' | 'food' | 'cooking' | 'cave'
    duration_days INTEGER NOT NULL,
    price_vnd INTEGER NOT NULL,
    available_seats INTEGER NOT NULL,
    departure_city TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hotels (
    hotel_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    star_rating INTEGER NOT NULL,
    price_per_night_vnd INTEGER NOT NULL,
    available_rooms INTEGER NOT NULL,
    breakfast_included INTEGER NOT NULL,   -- 0 | 1
    distance_to_centre_km REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS flights (
    flight_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL,        -- IATA code, e.g. HAN
    destination TEXT NOT NULL,
    airline TEXT NOT NULL,
    departure_time TEXT NOT NULL,    -- ISO time string
    duration_minutes INTEGER NOT NULL,
    price_vnd INTEGER NOT NULL,
    cabin_class TEXT NOT NULL,        -- 'economy' | 'business'
    seats_available INTEGER NOT NULL
);
"""


TOURS = [
    ("T001", "Halong Bay 2D1N Mid-range Cruise", "north", "cruise", 2, 5200000, 12, "Hanoi",
     "4-star Paradise Elegance-class overnight cruise. Includes kayaking, sunset, cooking class."),
    ("T002", "Halong Bay 3D2N Luxury Cruise", "north", "cruise", 3, 11500000, 4, "Hanoi",
     "5-star Stellar of the Seas. Lan Ha Bay extension, 2 nights onboard, jacuzzi suites."),
    ("T003", "Sapa 2D1N Homestay Trek", "north", "trek", 2, 3200000, 16, "Hanoi",
     "Sleeper bus to Sapa, Lao Chai → Ta Van trek, Hmong family homestay, return next evening."),
    ("T004", "Sapa 3D2N Fansipan Combo", "north", "trek", 3, 5800000, 8, "Hanoi",
     "2 nights Sapa town hotel, Fansipan cable car included, half-day Cat Cat village."),
    ("T005", "Hanoi Street Food Walking Tour", "north", "food", 1, 700000, 24, "Hanoi",
     "Evening 4-hour walk through Old Quarter, 6 tastings, English-speaking guide."),
    ("T006", "Hue Imperial City Day Tour", "central", "city", 1, 950000, 18, "Hue",
     "Imperial Citadel + Khai Dinh tomb + Minh Mang tomb, lunch included, English guide."),
    ("T007", "Hoi An Cooking Class with Market", "central", "cooking", 1, 850000, 14, "Hoi An",
     "Tra Que village morning market + 4-course cooking class, vegetarian option, certificate."),
    ("T008", "Phong Nha Paradise Cave Day Trip", "central", "cave", 1, 1500000, 10, "Dong Hoi",
     "1 km lit walkway through Paradise Cave + Dark Cave kayak, lunch included."),
    ("T009", "Cu Chi Tunnels Half-Day", "south", "city", 1, 650000, 30, "Ho Chi Minh City",
     "Morning or afternoon tour with English guide, includes tunnel crawl + tasting cassava."),
    ("T010", "Mekong Delta Overnight Homestay", "south", "city", 2, 2400000, 12, "Ho Chi Minh City",
     "Ben Tre coconut village, river boat + bicycle, homestay dinner, return next day."),
    ("T011", "Phu Quoc 3-Island Snorkel Day", "south", "city", 1, 850000, 20, "Phu Quoc",
     "South An Thoi archipelago: 3 islands, snorkel gear + lunch + transfer included."),
    ("T012", "HCMC Foodie Vespa Night Tour", "south", "food", 1, 1450000, 8, "Ho Chi Minh City",
     "4.5h pillion-passenger vespa tour, 5 food stops, 1 craft beer, local guide."),
]


HOTELS = [
    # Hanoi
    ("H001", "Hanoi Old Quarter Boutique", "Hanoi", 3, 850000, 6, 1, 0.4),
    ("H002", "Sofitel Legend Metropole Hanoi", "Hanoi", 5, 7800000, 3, 1, 0.5),
    ("H003", "Hanoi Backpacker Hostel Original", "Hanoi", 1, 200000, 24, 0, 0.6),
    # HCMC
    ("H004", "Liberty Central Saigon Citypoint", "Ho Chi Minh City", 4, 1900000, 10, 1, 0.3),
    ("H005", "Reverie Saigon", "Ho Chi Minh City", 5, 6500000, 5, 1, 0.2),
    ("H006", "Saigon Backpackers Pham Ngu Lao", "Ho Chi Minh City", 1, 220000, 30, 0, 0.4),
    # Hoi An
    ("H007", "Anantara Hoi An Resort", "Hoi An", 5, 5200000, 4, 1, 0.1),
    ("H008", "Hoi An Riverside Boutique", "Hoi An", 3, 950000, 8, 1, 0.7),
    ("H009", "An Bang Beach Bungalows", "Hoi An", 3, 1150000, 6, 0, 4.0),
    # Da Nang
    ("H010", "Furama Resort Da Nang", "Da Nang", 5, 4200000, 6, 1, 6.0),
    ("H011", "Brilliant Hotel Da Nang", "Da Nang", 4, 1450000, 12, 1, 0.5),
    # Sapa
    ("H012", "Sapa Highland Resort", "Sapa", 4, 1850000, 8, 1, 0.6),
    ("H013", "Topas Ecolodge Sapa", "Sapa", 4, 5500000, 4, 1, 18.0),
    ("H014", "Sapa Lake View Homestay", "Sapa", 2, 380000, 4, 0, 1.2),
    # Phu Quoc
    ("H015", "JW Marriott Phu Quoc Emerald Bay", "Phu Quoc", 5, 9800000, 6, 1, 12.0),
    ("H016", "Sunset Sanato Beach Club", "Phu Quoc", 4, 2200000, 8, 1, 8.0),
    ("H017", "Phu Quoc Ocean Pearl Hotel", "Phu Quoc", 3, 900000, 10, 1, 0.5),
]


FLIGHTS = [
    # HAN <-> SGN
    ("VN211", "HAN", "SGN", "Vietnam Airlines", "07:00", 130, 1850000, "economy", 45),
    ("VN221", "HAN", "SGN", "Vietnam Airlines", "13:00", 130, 2100000, "economy", 32),
    ("VJ131", "HAN", "SGN", "Vietjet Air", "06:30", 130, 950000, "economy", 60),
    ("VJ151", "HAN", "SGN", "Vietjet Air", "20:30", 135, 850000, "economy", 80),
    ("QH213", "HAN", "SGN", "Bamboo Airways", "09:00", 130, 1450000, "economy", 28),
    # HAN <-> DAD
    ("VN161", "HAN", "DAD", "Vietnam Airlines", "08:00", 85, 1250000, "economy", 40),
    ("VJ501", "HAN", "DAD", "Vietjet Air", "10:30", 90, 650000, "economy", 75),
    # SGN <-> DAD
    ("VN125", "SGN", "DAD", "Vietnam Airlines", "11:00", 85, 1350000, "economy", 38),
    ("VJ611", "SGN", "DAD", "Vietjet Air", "15:30", 85, 720000, "economy", 60),
    # SGN <-> PQC
    ("VJ921", "SGN", "PQC", "Vietjet Air", "08:15", 60, 750000, "economy", 90),
    ("VN191", "SGN", "PQC", "Vietnam Airlines", "12:45", 60, 1150000, "economy", 30),
    # HAN <-> PQC
    ("VJ451", "HAN", "PQC", "Vietjet Air", "06:00", 130, 1650000, "economy", 50),
    ("VN173", "HAN", "PQC", "Vietnam Airlines", "14:30", 130, 2450000, "economy", 25),
]


def seed(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO tours VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", TOURS
        )
        conn.executemany(
            "INSERT INTO hotels VALUES (?, ?, ?, ?, ?, ?, ?, ?)", HOTELS
        )
        conn.executemany(
            "INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", FLIGHTS
        )
        conn.commit()
    finally:
        conn.close()
    print(
        f"Seeded {len(TOURS)} tours, {len(HOTELS)} hotels, {len(FLIGHTS)} flights "
        f"into {db_path}"
    )


if __name__ == "__main__":
    seed()
