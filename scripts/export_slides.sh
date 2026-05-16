#!/usr/bin/env bash
# Export slides/presentation.md to both .pdf and .pptx using Marp CLI.
#
# Marp CLI ships its own Chromium for PDF, so the only prereq is npx
# (Node.js). The first run downloads Marp + Chromium (~250 MB) into
# npm cache; subsequent runs are fast.
#
# Usage:   bash scripts/export_slides.sh
# Output:  slides/presentation.pdf
#          slides/presentation.pptx

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/slides/presentation.md"

if ! command -v npx >/dev/null 2>&1; then
    echo "npx not found. Install Node.js 18+ from https://nodejs.org and retry."
    exit 1
fi

if [ ! -f "$SRC" ]; then
    echo "Source slides not found at $SRC"
    exit 1
fi

echo "[slides] Exporting PDF..."
npx --yes @marp-team/marp-cli@latest "$SRC" --pdf  --allow-local-files \
    --output "$ROOT/slides/presentation.pdf"

echo "[slides] Exporting PPTX..."
npx --yes @marp-team/marp-cli@latest "$SRC" --pptx --allow-local-files \
    --output "$ROOT/slides/presentation.pptx"

echo "[slides] Done."
ls -la "$ROOT/slides/"
