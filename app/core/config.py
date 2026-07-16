from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# ============================
# Project Paths
# ============================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"

MANUAL_DIR = DATA_DIR / "manuals"

VECTORSTORE_DIR = DATA_DIR / "vectorstore"

MANUAL_DIR.mkdir(parents=True, exist_ok=True)

VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# API Keys
# ============================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===========================
# Retry Configuration
# ===========================

MAX_RETRIES = 3

RETRY_DELAY = 2  # seconds

# ============================
# Models
# ============================

GEMINI_MODEL = "gemini/gemini-3.5-flash"
GROQ_MODEL = "groq/llama-3.1-8b-instant"

FALLBACK_CHAIN = [
    {
        "provider": "Gemini",
        "model": GEMINI_MODEL,
        "api_key": GEMINI_API_KEY
    },
    {
        "provider": "Groq",
        "model": GROQ_MODEL,
        "api_key": GROQ_API_KEY
    }
]
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ============================
# RAG
# ============================

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150

TOP_K = 3