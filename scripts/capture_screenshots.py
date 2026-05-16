"""Headless Playwright capture of the 5 required demo screenshots.

Prereqs: a valid AZURE_OPENAI_API_KEY (the assistant must call the real
LLM for the responses to be meaningful) and `playwright` installed:

    pip install playwright
    python -m playwright install chromium

Usage:

    python scripts/capture_screenshots.py

What it does:
1. Boots `streamlit run app.py` on port 8599 (avoids clashing with manual runs).
2. Opens the page in Chromium, pastes the API key, clicks Boot.
3. Walks through 5 scenarios from sample_queries.json (S01, S03, S05, S06, S08).
4. Saves PNGs to docs/screenshots/01..05*.png.
5. Tears down the Streamlit subprocess.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOT_DIR = ROOT / "docs" / "screenshots"

SCENARIOS = [
    # (filename, prompt or None for initial state, extra_action)
    ("01-initial-state.png", None, None),
    ("02-rag-vietnamese.png", "Mình có 3 ngày ở Hà Nội thì nên đi đâu, ăn gì?", None),
    ("03-tool-call-expanded.png", None, "expand_last_trace"),
    ("04-multi-tool-budget.png",
     "Plan a 7-day Vietnam itinerary for 2 people, budget USD 1,500 total. "
     "Show me a sample plan and the budget calculation.", None),
    ("05-out-of-domain-redirect.png",
     "What's the stock price of Apple today?", None),
]


def _free_port(default: int = 8599) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _wait_for_streamlit(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                time.sleep(1.0)  # Let Streamlit finish booting
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not start on port {port} within {timeout}s")


def main() -> int:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        print(
            "AZURE_OPENAI_API_KEY missing — the assistant cannot generate "
            "real responses without it. Aborting.",
            file=sys.stderr,
        )
        return 2

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright not installed. Install with:\n"
            "    pip install playwright\n"
            "    python -m playwright install chromium",
            file=sys.stderr,
        )
        return 2

    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    print(f"[capture] booting Streamlit on port {port}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ},
    )
    try:
        _wait_for_streamlit(port)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
            page.wait_for_timeout(1500)

            # Sidebar: paste API key + click Boot.
            key_input = page.get_by_label("Azure OpenAI API key")
            key_input.fill(api_key)
            page.get_by_role("button", name="Boot / Reload agent").click()
            # Wait for green "Agent ready." badge.
            page.get_by_text("Agent ready.").wait_for(timeout=60_000)
            page.wait_for_timeout(500)

            for filename, prompt, extra in SCENARIOS:
                if prompt:
                    chat_input = page.locator(
                        'textarea[data-testid="stChatInputTextArea"]'
                    )
                    chat_input.fill(prompt)
                    chat_input.press("Enter")
                    # Wait for the next assistant turn to finish: the Tool
                    # calls expander appears (or 60s timeout for the no-tool
                    # case S08).
                    try:
                        page.locator(":text-matches('Tool calls', 'i')").last.wait_for(
                            timeout=90_000
                        )
                    except Exception:
                        page.wait_for_timeout(8000)
                    page.wait_for_timeout(2500)

                if extra == "expand_last_trace":
                    # Click the most recent "Tool calls" expander.
                    expanders = page.locator(":text-matches('Tool calls', 'i')")
                    if expanders.count() > 0:
                        expanders.last.click()
                        page.wait_for_timeout(800)

                target = SHOT_DIR / filename
                page.screenshot(path=str(target), full_page=True)
                print(f"[capture] saved {target.relative_to(ROOT)}")

            browser.close()
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
