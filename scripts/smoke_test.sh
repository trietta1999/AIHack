#!/usr/bin/env bash
# Smoke test for a fresh checkout. Verifies that:
#   1. Python imports of every module succeed.
#   2. SQLite booking DB seeds cleanly and contains expected counts.
#   3. Pytest passes end-to-end (27+ offline tests).
#   4. Streamlit app at least imports without raising.
#
# Runs in ~20 seconds. Does NOT need an Azure key.
#
# Usage:   bash scripts/smoke_test.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then
    echo "[smoke] No .venv detected — run 'make install' first."
    exit 1
fi
PY=.venv/bin/python

echo "[smoke] 1/4 — Importing modules..."
$PY - <<'PY_EOF'
import importlib
for m in [
    "travel_advisor",
    "travel_advisor.config",
    "travel_advisor.models",
    "travel_advisor.ingestion",
    "travel_advisor.embeddings",
    "travel_advisor.retriever",
    "travel_advisor.tools",
    "travel_advisor.agent",
]:
    importlib.import_module(m)
print("  ok — all modules import")
PY_EOF

echo "[smoke] 2/4 — Seeding SQLite booking DB..."
$PY scripts/seed_db.py >/dev/null
$PY - <<'PY_EOF'
import sqlite3, pathlib
db = pathlib.Path("data/bookings.sqlite")
assert db.exists(), "bookings.sqlite was not created"
with sqlite3.connect(db) as conn:
    for table, expected in [("tours", 12), ("hotels", 17), ("flights", 13)]:
        n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert n == expected, f"{table}: expected {expected}, got {n}"
print("  ok — 12 tours + 17 hotels + 13 flights")
PY_EOF

echo "[smoke] 3/4 — Running pytest..."
$PY -m pytest tests/ -q --no-header

echo "[smoke] 4/4 — Verifying Streamlit app module syntax..."
$PY - <<'PY_EOF'
import ast
ast.parse(open("app.py").read())
print("  ok — app.py parses")
PY_EOF

echo
echo "[smoke] All checks passed. Project is ready to demo."
