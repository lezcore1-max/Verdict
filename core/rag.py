"""
core/rag.py — ChromaDB vector store for paper chunk retrieval.

Pre-ingestion step: all papers are chunked via pdf_parser and stored here
before claim processing begins.  Evidence hunters query this store alongside
Tavily to find relevant prior work.
"""
import logging
import os

import chromadb
from chromadb.config import Settings

from core.config import CHROMA_DIR, EMBED_MODEL
from core.embedder import ChromaEmbeddingFunction

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "verdict_papers"


def get_chroma_collection(persist_dir: str = CHROMA_DIR) -> chromadb.Collection:
    """
    Create or load the ChromaDB collection.
    Uses a custom EmbeddingFunction that wraps the local sentence-transformer.
    """
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    embed_fn = ChromaEmbeddingFunction(model_name=EMBED_MODEL)
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest_chunks(
    collection: chromadb.Collection,
    paper_id: int,
    chunks: list[str],
) -> None:
    """
    Add text chunks to the collection.  Each chunk is stored with paper_id
    and chunk_index as metadata.  Already-ingested papers are detected by the
    caller via database.is_paper_ingested — this function does not check.
    """
    if not chunks:
        return

    ids = [f"paper_{paper_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"paper_id": paper_id, "chunk_index": i} for i in range(len(chunks))]

    # ChromaDB upsert silently ignores existing IDs
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info("Ingested %d chunks for paper_id=%d", len(chunks), paper_id)


def query_rag(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
    paper_id_exclude: int | None = None,
) -> list[dict]:
    """
    Query the collection for chunks most similar to query_text.

    Returns a list of dicts:
        {"content": str, "paper_id": int, "chunk_index": int, "distance": float}

    Optionally exclude chunks from a specific paper (the paper being analysed)
    to avoid the model seeing its own text as evidence.
    """
    try:
        where = None
        if paper_id_exclude is not None:
            where = {"paper_id": {"$ne": paper_id_exclude}}

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append(
                {
                    "content": doc,
                    "paper_id": meta.get("paper_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "distance": dist,
                }
            )
        return output
    except Exception as exc:
        logger.warning("ChromaDB query failed: %s", exc)
        return []
