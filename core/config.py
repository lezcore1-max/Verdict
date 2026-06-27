"""core/config.py — Environment variable loading and global constants."""
import os
import json
from dotenv import load_dotenv

# Load .env from the verdict/ directory (two levels up from this file)
_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(_HERE)
load_dotenv(os.path.join(_ROOT, ".env"), override=False)

# ── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DECOMPOSER_MODEL: str = os.getenv("DECOMPOSER_MODEL", "gemini-2.0-flash")
SYNTHESIZER_MODEL: str = os.getenv("SYNTHESIZER_MODEL", "gemini-2.0-flash")

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Tavily ───────────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
TAVILY_QUERY_CAP: int = int(os.getenv("TAVILY_QUERY_CAP", "200"))

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_DIR: str = os.getenv("CHROMA_DIR", os.path.join(_ROOT, "chroma_db"))

# ── SPRT thresholds (Wald: α=0.05, β=0.05) ────────────────────────────────────
UPPER_THRESHOLD: float = 19.0     # (1 - β) / α  = 0.95 / 0.05 = 19.0
LOWER_THRESHOLD: float = 0.0526   # β / (1 - α)  = 0.05 / 0.95 ≈ 0.0526
P_VALUE_FLOOR: float = 1e-6       # Clamp minimum to prevent division by zero

# ── Dempster-Shafer ───────────────────────────────────────────────────────────
K_CONFLICT: float = 0.1           # Below this → ConflictError

# ── Deduplication thresholds ──────────────────────────────────────────────────
COSINE_DEDUP: float = 0.85        # Evidence deduplication
CLAIM_DEDUP: float = 0.85         # Claim deduplication across chunks

# ── Papers With Code metric alias map ─────────────────────────────────────────
METRIC_ALIASES: dict[str, str] = {
    "accuracy": "acc",
    "f1-score": "f1",
    "f1 score": "f1",
    "bleu": "bleu",
    "top-1": "top_1_acc",
    "top-1 accuracy": "top_1_acc",
    "rouge-l": "rougeL",
    "rouge l": "rougeL",
    "map": "mean_average_precision",
    "mean average precision": "mean_average_precision",
}

# Allow user extension via env var
_extra_raw = os.getenv("METRIC_ALIASES_EXTRA", "")
if _extra_raw.strip():
    try:
        _extra = json.loads(_extra_raw)
        METRIC_ALIASES.update(_extra)
    except json.JSONDecodeError:
        pass  # Silently ignore malformed JSON
