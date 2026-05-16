# Vietnam Travel Planner — single-stage image, ~600 MB.
#
# Build:   docker build -t vn-travel-planner .
# Run:     docker run --rm -p 8501:8501 \
#            -e AZURE_OPENAI_API_KEY=... \
#            -v "$(pwd)/data:/app/data" \
#            vn-travel-planner
#
# The container builds the FAISS index + seeds the SQLite DB on first
# boot (idempotent), then serves the Streamlit app on port 8501.

FROM python:3.12-slim

# Faster, smaller installs.
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS deps for faiss-cpu / streamlit.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so layer cache survives source edits.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source.
COPY . .

EXPOSE 8501

# Healthcheck hits Streamlit's built-in liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" \
      || exit 1

# Boot script: seed DB + build FAISS index (idempotent), then serve.
CMD ["/bin/sh", "-c", "\
  if [ ! -f data/bookings.sqlite ]; then python scripts/seed_db.py; fi && \
  if [ ! -f data/faiss_index/index.faiss ]; then \
    if [ -z \"$AZURE_OPENAI_API_KEY\" ]; then \
      echo 'AZURE_OPENAI_API_KEY missing — set it via -e to build the index'; exit 1; \
    fi; \
    python scripts/build_index.py; \
  fi && \
  streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true \
"]
