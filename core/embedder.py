"""
core/embedder.py — Local sentence-transformer embeddings.

Singleton pattern: the model is loaded once per process.
All vectors are L2-normalised so cosine similarity == dot product.
"""
import threading
import numpy as np
from typing import Callable, List

from core.config import EMBED_MODEL

# ChromaDB EmbeddingFunction import (optional; only needed if chromadb installed)
try:
    from chromadb import EmbeddingFunction, Documents, Embeddings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class LocalEmbedder:
    """Thread-safe singleton wrapping sentence-transformers."""

    _instance: "LocalEmbedder | None" = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = EMBED_MODEL) -> "LocalEmbedder":
        with cls._lock:
            if cls._instance is None or cls._instance._model_name != model_name:
                obj = super().__new__(cls)
                obj._model_name = model_name
                obj._model = None  # lazy load
                cls._instance = obj
        return cls._instance

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised vectors.
        Shape: (len(texts), embedding_dim).
        """
        self._load()
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Dot product of two L2-normalised vectors == cosine similarity.
        Vectors may be 1-D (single embedding) or 2-D (batch).
        """
        return float(np.dot(a.flatten(), b.flatten()))

    def deduplicate(
        self,
        items: list,
        key_fn: Callable,
        threshold: float = 0.85,
        exact_key_fn: Callable | None = None,
    ) -> list:
        """
        Remove duplicates from `items`.

        Args:
            items:         List of arbitrary objects.
            key_fn:        Callable(item) -> str  — text to embed for comparison.
            threshold:     Cosine similarity threshold above which items are duplicates.
            exact_key_fn:  Optional callable(item) -> str for exact-string pre-check.
                           If two items share the same exact key, one is dropped immediately
                           without embedding comparison.

        Returns:
            Deduplicated list (preserving first occurrence).
        """
        if not items:
            return []

        kept: list = []
        kept_texts: list[str] = []
        kept_vecs: list[np.ndarray] = []
        exact_seen: set[str] = set()

        for item in items:
            text = key_fn(item)

            # Exact-string pre-check (O(1))
            if exact_key_fn is not None:
                exact_key = exact_key_fn(item).strip().lower()
                if exact_key in exact_seen:
                    continue
                exact_seen.add(exact_key)

            # Embedding similarity check
            vec = self.encode([text])[0]
            is_dup = False
            for kv in kept_vecs:
                if self.cosine_similarity(vec, kv) >= threshold:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(item)
                kept_texts.append(text)
                kept_vecs.append(vec)

        return kept


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB EmbeddingFunction wrapper
# ─────────────────────────────────────────────────────────────────────────────

if _CHROMA_AVAILABLE:
    class ChromaEmbeddingFunction(EmbeddingFunction):
        """Wraps LocalEmbedder so ChromaDB can call it internally."""

        def __init__(self, model_name: str = EMBED_MODEL) -> None:
            self._embedder = LocalEmbedder(model_name)

        def __call__(self, input: Documents) -> Embeddings:
            vecs = self._embedder.encode(list(input))
            return vecs.tolist()

else:
    class ChromaEmbeddingFunction:  # type: ignore[no-redef]
        """Stub when chromadb is not installed."""
        def __init__(self, model_name: str = EMBED_MODEL) -> None:
            self._embedder = LocalEmbedder(model_name)

        def __call__(self, input):
            vecs = self._embedder.encode(list(input))
            return vecs.tolist()
