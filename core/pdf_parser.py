"""
core/pdf_parser.py — PyMuPDF-based PDF text extraction and chunking.

Multi-column layout handling: text blocks are sorted by (column_index, y0)
where column_index = x0 // column_width.  This ensures left-column text
precedes right-column text rather than interleaving by y-coordinate.
"""
import re
import logging
from typing import List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Approximate characters per token (whitespace-split approximation)
_CHARS_PER_TOKEN = 4
_COLUMN_WIDTH_PT = 300  # Points; assumes two-column layout with ~300pt columns


# ─────────────────────────────────────────────────────────────────────────────
# Section keywords for epistemic weight tagging
# ─────────────────────────────────────────────────────────────────────────────

_HIGH_WEIGHT_SECTIONS = {"abstract", "results", "experiments", "evaluation"}
_LOW_WEIGHT_SECTIONS = {"discussion", "future work", "limitations", "conclusion"}

# Hedging language patterns
_HIGH_HEDGE_WORDS = re.compile(
    r'\b(demonstrates?|proves?|shows?|establishes?|confirms?)\b', re.I
)
_LOW_HEDGE_WORDS = re.compile(
    r'\b(suggests?|may\s+indicate|might|could|appears?\s+to|seems?\s+to)\b', re.I
)


def extract_text(pdf_path: str) -> str:
    """
    Extract full text from a PDF file.

    Multi-column handling:
      - For each page, sort text blocks by (column_index, y0).
      - column_index = int(x0 // COLUMN_WIDTH_PT)
      - This produces correct reading order for standard two-column papers.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Cannot open PDF %s: %s", pdf_path, exc)
        return ""

    pages_text = []
    for page in doc:
        blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,block_no,block_type)
        # Filter to text blocks (block_type == 0)
        text_blocks = [b for b in blocks if b[6] == 0]
        # Sort by column then y position
        text_blocks.sort(key=lambda b: (int(b[0] // _COLUMN_WIDTH_PT), b[1]))
        page_text = " ".join(b[4].strip() for b in text_blocks if b[4].strip())
        pages_text.append(page_text)

    doc.close()
    return "\n\n".join(pages_text)


def detect_section(text_span: str) -> str:
    """
    Heuristic section detector.  Looks for section header keywords in the
    first 200 chars of a text span.  Returns a normalised section name.
    """
    header = text_span[:200].lower()
    if "abstract" in header:
        return "abstract"
    if any(k in header for k in ("result", "experiment", "evaluation", "benchmark")):
        return "results"
    if any(k in header for k in ("introduc",)):
        return "introduction"
    if any(k in header for k in ("discussion", "future", "limitation")):
        return "discussion"
    if "conclusion" in header:
        return "conclusion"
    if any(k in header for k in ("method", "approach", "model", "architecture")):
        return "methods"
    return "body"


def epistemic_weight_for_section(section: str) -> float:
    """Higher weight for claims from results/abstract; lower for discussion."""
    s = section.lower()
    if s in _HIGH_WEIGHT_SECTIONS or s == "abstract":
        return 0.85
    if s in _LOW_WEIGHT_SECTIONS:
        return 0.40
    return 0.60


def hedge_adjustment(claim_text: str) -> float:
    """
    Return a multiplier in (0.8, 1.0] based on hedging language.
    High-confidence verbs → 1.0; hedging verbs → 0.8.
    """
    if _HIGH_HEDGE_WORDS.search(claim_text):
        return 1.0
    if _LOW_HEDGE_WORDS.search(claim_text):
        return 0.8
    return 0.9


def chunk_text(
    text: str,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Tokenisation approximation: split on whitespace (1 token ≈ 4 chars used
    only for chunk-size estimation; actual split is on words).
    """
    words = text.split()
    chunk_words = chunk_tokens  # 1 word ≈ 1 token (good enough for chunking)
    overlap_words = overlap_tokens
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


def chunk_for_agent1(
    text: str,
    chunk_tokens: int = 4000,
    overlap_tokens: int = 500,
) -> List[str]:
    """Larger chunks for Agent 1 context window."""
    return chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
