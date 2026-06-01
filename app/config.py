"""Central configuration.

Loads environment variables from a `.env` file located at the project root,
regardless of which directory you launched `streamlit run` from.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root (the folder that contains `main.py` and `.env`)
# as an absolute path — this is the fix for "GROQ_API_KEY not in env file"
# when Streamlit is launched from a different working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# `override=False` so real OS env vars win over the .env file.
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _clean(value: str | None) -> str:
    """Strip whitespace and surrounding quotes that often sneak into .env files."""
    if not value:
        return ""
    return value.strip().strip('"').strip("'")


GROQ_API_KEY = _clean(os.getenv("GROQ_API_KEY"))
print("=" * 50)
print("CONFIG LOADED")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("ENV_PATH:", ENV_PATH)
print("ENV EXISTS:", ENV_PATH.exists())
print("GROQ_API_KEY:", bool(GROQ_API_KEY))
print("=" * 50)
TAVILY_API_KEY = _clean(os.getenv("TAVILY_API_KEY"))
GROQ_MODEL = _clean(os.getenv("GROQ_MODEL")) or "llama-3.1-8b-instant"

# Embedding model — small, fast, runs locally on CPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Where the FAISS vector index is persisted on disk (relative to project root).
VECTORSTORE_DIR = str(PROJECT_ROOT / "data" / "vectorstore")

# Chunking parameters: balances context and retrieval precision.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# How many chunks to retrieve per query.
TOP_K = 4


def assert_groq_key() -> None:
    """Raise a clear error if the Groq key isn't configured."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            f"GROQ_API_KEY not found.\n"
            f"Looked for it in: {ENV_PATH}\n"
            f"Fix: create that file and add a line like:\n"
            f"    GROQ_API_KEY=gsk_your_real_key_here\n"
            f"Then fully restart the Streamlit server (Ctrl+C and re-run)."
        )
