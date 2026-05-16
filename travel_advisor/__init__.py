"""Vietnam Travel Planner — Hackathon RAG + agent package.

Top-level imports expose only the public symbols needed by callers (CLI,
Streamlit, tests). Internal modules are still importable via their full
path when needed.
"""

from .config import settings
from .models import Chunk, RetrievedChunk

__all__ = ["settings", "Chunk", "RetrievedChunk"]
