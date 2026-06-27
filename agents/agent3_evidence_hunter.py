"""
agents/agent3_evidence_hunter.py — Retrieve supporting evidence for a sub-hypothesis.

Sources (in order):
  1. Tavily search API (async, multiple queries)
  2. Papers With Code API (benchmark claims only)
  3. ChromaDB RAG store (local prior paper chunks)

Deduplication runs on the combined result set before returning.
"""
import asyncio
import logging
import re
import string
from typing import Optional

import httpx

from core.config import (
    GEMINI_MODEL, TAVILY_API_KEY, TAVILY_QUERY_CAP, COSINE_DEDUP, METRIC_ALIASES
)
from core.gemini_client import GeminiClient
from core.embedder import LocalEmbedder
from agents.schemas import EvidenceItem, EvidenceOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an evidence retrieval specialist. Given a sub-hypothesis from a machine learning research paper, generate 2-3 targeted search queries that would find external evidence to test it.

For each query, think about:
- What datasets, benchmarks, or experimental setups would directly test this claim?
- What papers would have studied this specific phenomenon?
- What metrics or results would constitute evidence?

Return a JSON object with:
{
  "queries": ["query 1", "query 2", "query 3"]
}

Respond ONLY with a valid JSON object. No markdown fences, no explanation, no preamble."""

_TAVILY_URL = "https://api.tavily.com/search"
_PWC_BASE = "https://paperswithcode.com/api/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run(
    sub_hyp_text: str,
    claim_type: str,
    chroma_collection,
    paper_id: int,
    db_conn,
    model_name: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
    tavily_key: Optional[str] = None,
) -> EvidenceOutput:
    """
    Retrieve evidence for a sub-hypothesis.

    Returns EvidenceOutput with deduplicated evidence items.
    """
    tavily_api_key = tavily_key or TAVILY_API_KEY
    queries = _generate_queries(sub_hyp_text, model_name, api_key)

    evidence_items: list[EvidenceItem] = []

    # 1. Tavily async search
    tavily_results = asyncio.run(
        _tavily_search_all(queries, tavily_api_key, paper_id, db_conn)
    )
    evidence_items.extend(tavily_results)

    # 2. Papers With Code (benchmark claims only)
    if claim_type == "benchmark_performance":
        pwc_results = _pwc_search(sub_hyp_text)
        evidence_items.extend(pwc_results)

    # 3. ChromaDB RAG
    if chroma_collection is not None:
        rag_results = _rag_search(sub_hyp_text, chroma_collection, paper_id)
        evidence_items.extend(rag_results)

    # Deduplicate across all sources
    embedder = LocalEmbedder()
    deduped = embedder.deduplicate(
        evidence_items,
        key_fn=lambda e: e.content,
        threshold=COSINE_DEDUP,
        exact_key_fn=_doi_arxiv_key,
    )

    logger.info(
        "Agent 3: %d evidence items after dedup (from %d raw)",
        len(deduped), len(evidence_items)
    )
    return EvidenceOutput(evidence=deduped)


# ─────────────────────────────────────────────────────────────────────────────
# Query generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_queries(
    sub_hyp_text: str,
    model_name: str,
    api_key: Optional[str],
) -> list[str]:
    client = GeminiClient(
        model_name=model_name,
        temperature=0.2,
        system_prompt=_SYSTEM_PROMPT,
        api_key=api_key,
    )
    raw = client.call(f"Sub-hypothesis: {sub_hyp_text}")
    if raw and "queries" in raw:
        return [str(q) for q in raw["queries"][:3]]
    # Fallback: use the sub-hypothesis text itself as a query
    return [sub_hyp_text[:200]]


# ─────────────────────────────────────────────────────────────────────────────
# Tavily async search
# ─────────────────────────────────────────────────────────────────────────────

async def _tavily_search_all(
    queries: list[str],
    api_key: str,
    paper_id: int,
    db_conn,
) -> list[EvidenceItem]:
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — skipping Tavily search")
        return []

    tasks = []
    for q in queries:
        # Check and increment counter atomically BEFORE spawning the task
        from core.database import atomic_increment_tavily_counter
        allowed = atomic_increment_tavily_counter(db_conn, paper_id, TAVILY_QUERY_CAP)
        if not allowed:
            logger.info("Tavily query cap (%d) reached — skipping: %s", TAVILY_QUERY_CAP, q[:60])
            continue
        tasks.append(_tavily_single(q, api_key))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[EvidenceItem] = []
    for r in results:
        if isinstance(r, list):
            items.extend(r)
        elif isinstance(r, Exception):
            logger.warning("Tavily search error: %s", r)
    return items


async def _tavily_single(query: str, api_key: str) -> list[EvidenceItem]:
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": 5,
        "include_raw_content": True,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(_TAVILY_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    items = []
    for result in data.get("results", []):
        url = result.get("url", "")
        content = result.get("content", "")
        raw = result.get("raw_content", content)
        tier = _infer_reliability_tier(url)
        items.append(EvidenceItem(
            source=url,
            content=content[:1000],
            reliability_tier=tier,
            directness="partial_test",
            raw_content=raw[:3000],
            agent_source="agent3",
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Papers With Code
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_metric(metric: str) -> str:
    """Normalise a metric name using the alias map."""
    cleaned = metric.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    return METRIC_ALIASES.get(cleaned, METRIC_ALIASES.get(metric.lower(), cleaned))


def get_pwc_leaderboard(task: str, dataset: str, metric: str) -> Optional[list[dict]]:
    """Fetch raw leaderboard array for formal t-tests."""
    if not (task or dataset) or not metric:
        return None

    try:
        with httpx.Client(timeout=20) as client:
            params = {}
            if task: params["task"] = task
            if dataset: params["dataset"] = dataset
            params["metric"] = _normalise_metric(metric)

            resp = client.get(f"{_PWC_BASE}/results/", params=params)
            if resp.status_code != 200 or not resp.json().get("results"):
                if task:
                    resp = client.get(f"{_PWC_BASE}/sota/", params={"task": task})
                else:
                    return None
            
            data = resp.json()
            results = data.get("results", [])
            # Extract just the score values for the specified metric
            leaderboard = []
            norm_metric = _normalise_metric(metric)
            for r in results:
                metrics_dict = r.get("metrics", {})
                # Attempt to find the metric (case-insensitive)
                for k, v in metrics_dict.items():
                    if _normalise_metric(k) == norm_metric or k.lower() == metric.lower():
                        try:
                            # some PwC scores are strings like "98.2%"
                            val = float(str(v).replace('%', ''))
                            leaderboard.append({"score": val})
                        except (ValueError, TypeError):
                            pass
                        break
            
            return leaderboard if leaderboard else None
    except Exception as exc:
        logger.warning("Papers With Code leaderboard query failed: %s", exc)
        return None


def _pwc_search(sub_hyp_text: str) -> list[EvidenceItem]:
    """Query Papers With Code for leaderboard data related to the sub-hypothesis."""
    # Extract likely metric and dataset from sub-hypothesis text
    # (heuristic: look for quoted words or known metric names)
    metric_guess = _extract_metric_guess(sub_hyp_text)
    dataset_guess = _extract_dataset_guess(sub_hyp_text)
    task_guess = _extract_task_guess(sub_hyp_text)

    items: list[EvidenceItem] = []

    try:
        with httpx.Client(timeout=20) as client:
            # Primary: /api/results/ with task, dataset, metric
            params = {}
            if task_guess:
                params["task"] = task_guess
            if dataset_guess:
                params["dataset"] = dataset_guess
            if metric_guess:
                params["metric"] = _normalise_metric(metric_guess)

            resp = client.get(f"{_PWC_BASE}/results/", params=params)

            if resp.status_code != 200 or not resp.json().get("results"):
                # Fallback: /api/sota/ with task only
                if task_guess:
                    resp = client.get(
                        f"{_PWC_BASE}/sota/",
                        params={"task": task_guess}
                    )
                else:
                    return []

            data = resp.json()
            results_list = data.get("results", [])[:5]
            for r in results_list:
                model = r.get("model_name", "unknown")
                score = r.get("metrics", {})
                content = f"PwC leaderboard: {model} — {score}"
                items.append(EvidenceItem(
                    source=f"paperswithcode.com (task={task_guess})",
                    content=content,
                    reliability_tier="peer_reviewed",
                    directness="direct_test",
                    raw_content=str(r),
                    agent_source="agent3",
                ))
    except Exception as exc:
        logger.warning("Papers With Code query failed: %s", exc)

    return items


def _extract_metric_guess(text: str) -> str:
    known = list(METRIC_ALIASES.keys())
    text_lower = text.lower()
    for m in known:
        if m in text_lower:
            return m
    return ""


def _extract_dataset_guess(text: str) -> str:
    datasets = ["imagenet", "cifar", "coco", "glue", "superglue", "squad",
                "ms-coco", "voc", "ade20k", "wmt", "conll"]
    text_lower = text.lower()
    for d in datasets:
        if d in text_lower:
            return d
    return ""


def _extract_task_guess(text: str) -> str:
    tasks = {
        "image classification": ["image classification", "classify images"],
        "object detection": ["object detection", "detect objects"],
        "semantic segmentation": ["segmentation"],
        "machine translation": ["translation", "translate"],
        "question answering": ["question answering", "reading comprehension"],
        "text classification": ["text classification", "sentiment"],
        "named entity recognition": ["ner", "named entity"],
    }
    text_lower = text.lower()
    for task, keywords in tasks.items():
        if any(k in text_lower for k in keywords):
            return task
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB RAG search
# ─────────────────────────────────────────────────────────────────────────────

def _rag_search(
    sub_hyp_text: str,
    chroma_collection,
    paper_id: int,
) -> list[EvidenceItem]:
    from core.rag import query_rag
    chunks = query_rag(chroma_collection, sub_hyp_text, n_results=5, paper_id_exclude=paper_id)
    items = []
    for chunk in chunks:
        items.append(EvidenceItem(
            source=f"RAG:paper_{chunk.get('paper_id','?')}_chunk_{chunk.get('chunk_index','?')}",
            content=chunk["content"][:500],
            reliability_tier="preprint",
            directness="partial_test",
            raw_content=chunk["content"],
            agent_source="agent3",
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_reliability_tier(url: str) -> str:
    url_lower = url.lower()
    if any(d in url_lower for d in ["arxiv.org", "openreview.net", "aclanthology.org",
                                     "proceedings.mlr.press", "neurips.cc", "icml.cc",
                                     "iclr.cc", "aaai.org", "acm.org", "ieee.org"]):
        return "preprint"
    if any(d in url_lower for d in ["nature.com", "science.org", "springer.com",
                                     "wiley.com", "elsevier.com"]):
        return "peer_reviewed"
    return "blog"


def _doi_arxiv_key(item: EvidenceItem) -> str:
    """Build a dedup key from DOI or arxiv ID, or fall back to source URL."""
    if item.doi:
        return f"doi:{item.doi}"
    if item.arxiv_id:
        return f"arxiv:{item.arxiv_id}"
    return item.source


def get_pwc_leaderboard(sub_hyp_text: str, metric: str, dataset: str) -> list[dict]:
    """
    Fetch top-20 scores from PwC for a specific metric+dataset combination.
    Used by Agent 4 for formal t-test computation.
    Returns list of dicts with 'score' key.
    """
    scores = []
    try:
        with httpx.Client(timeout=20) as client:
            normalised = _normalise_metric(metric)
            resp = client.get(
                f"{_PWC_BASE}/results/",
                params={"dataset": dataset, "metric": normalised},
            )
            if resp.status_code == 200:
                for r in resp.json().get("results", [])[:20]:
                    m = r.get("metrics", {})
                    val = m.get(normalised) or m.get(metric)
                    if val is not None:
                        try:
                            scores.append({"score": float(val), "model": r.get("model_name", "")})
                        except (TypeError, ValueError):
                            pass
    except Exception as exc:
        logger.warning("PwC leaderboard fetch failed: %s", exc)
    return scores
