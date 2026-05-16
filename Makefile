# Vietnam Travel Planner — convenience targets.
#
# Common usage:
#   make install     create .venv + install requirements
#   make seed        seed SQLite bookings DB
#   make index       build FAISS index from KB (needs Azure key)
#   make test        run the pytest suite (offline, no key)
#   make demo        seed + index + launch Streamlit
#   make docker      build and run the Docker image
#   make slides      export slides to PDF + PPTX via Marp
#   make screenshots automated headless capture of 5 demo screens
#   make smoke       fresh-checkout smoke test
#   make clean       remove build artefacts (keeps .venv)
#
# Override the Python interpreter:   make demo PYTHON=python3.11

PYTHON ?= python3.12
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python
ST     := $(VENV)/bin/streamlit

.PHONY: install seed index test demo docker docker-build docker-run slides screenshots smoke clean

install: $(VENV)/bin/activate
$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

seed: install
	$(PY) scripts/seed_db.py

index: install
	$(PY) scripts/build_index.py

test: install
	$(PY) -m pytest tests/ -v

demo: install seed
	@if [ ! -f data/faiss_index/index.faiss ]; then \
		echo ">>> FAISS index missing, building (needs OPENAI_API_KEY)..."; \
		$(PY) scripts/build_index.py; \
	fi
	$(ST) run app.py

docker: docker-build docker-run

docker-build:
	docker build -t vn-travel-planner .

docker-run:
	docker run --rm -p 8501:8501 \
		-e OPENAI_API_KEY="$$OPENAI_API_KEY" \
		-e TAVILY_API_KEY="$$TAVILY_API_KEY" \
		-v "$$(pwd)/data:/app/data" \
		vn-travel-planner

slides: install
	@command -v marp >/dev/null 2>&1 || { \
		echo "marp CLI not found. Install with: npm install -g @marp-team/marp-cli"; \
		exit 1; \
	}
	marp slides/presentation.md --pdf  --output slides/presentation.pdf
	marp slides/presentation.md --pptx --output slides/presentation.pptx
	@echo ">>> Exported slides/presentation.{pdf,pptx}"

screenshots: install seed
	@if [ ! -f data/faiss_index/index.faiss ]; then \
		echo "FAISS index missing — run 'make index' first or set up demo mode." ; exit 1; \
	fi
	$(PIP) install playwright >/dev/null
	$(PY) -m playwright install chromium
	$(PY) scripts/capture_screenshots.py

smoke: install
	bash scripts/smoke_test.sh

clean:
	rm -rf data/faiss_index data/bookings.sqlite .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo ">>> Cleaned build artefacts. Run 'make seed index' to rebuild."
