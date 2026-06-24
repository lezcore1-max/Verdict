"""
agents/agent5_devils_advocate.py — Generate adversarial counterevidence.

Temperature: 0.7 (higher than other agents to encourage creative adversarial queries).
Same EvidenceOutput structure as Agent 3.
"""
import asyncio
import logging
from typing import Optional

import httpx

from core.config import GEMINI_MODEL, TAVILY_API_KEY, TAVILY_QUERY_CAP, COSINE_DEDUP
from core.gemini_client import GeminiClient
from core.embedder import LocalEmbedder
from agents.schemas import EvidenceItem, EvidenceOutput
from agents.agent3_evidence_hunter import _infer_reliability_tier, _doi_arxiv_key

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a scientific devil's advocate. Your role is to generate adversarial search queries that target COUNTEREVIDENCE against a given sub-hypothesis.

You are looking for:
- Failed replications of similar claims
- Results that contradict or limit the scope of the claim
- Boundary conditions where the claim breaks down
- Methodological criticisms that would invalidate the claim
- Papers that show the opposite result under similar conditions

Generate 2-3 adversarial search queries designed to find the strongest counterevidence.

Return a JSON object with:
{
  "queries": ["adversarial query 1", "adversarial query 2", "adversarial query 3"]
}

Be creative and aggressive. Your goal is to falsify the sub-hypothesis.

Respond ONLY with a valid JSON object. No markdown fences, no explanation, no preamble."""

_TAVILY_URL = "https://api.tavily.com/search"


def run(
    sub_hyp_text: str,
    paper_id: int,
    db_conn,
    model_name: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
    tavily_key: Optional[str] = None,
) -> EvidenceOutput:
    """
    Generate adversarial queries and retrieve counterevidence.

    Returns EvidenceOutput with deduplicated counterevidence items.
    """
    tavily_api_key = tavily_key or TAVILY_API_KEY
    queries = _generate_adversarial_queries(sub_hyp_text, model_name, api_key)

    evidence_items = asyncio.run(
        _tavily_search_adversarial(queries, tavily_api_key, paper_id, db_conn)
    )

    # Deduplicate
    embedder = LocalEmbedder()
    deduped = embedder.deduplicate(
        evidence_items,
        key_fn=lambda e: e.content,
        threshold=COSINE_DEDUP,
        exact_key_fn=_doi_arxiv_key,
    )

    logger.info(
        "Agent 5: %d adversarial evidence items after dedup (from %d raw)",
        len(deduped), len(evidence_items)
    )
    return EvidenceOutput(evidence=deduped)


def _generate_adversarial_queries(
    sub_hyp_text: str,
    model_name: str,
    api_key: Optional[str],
) -> list[str]:
    client = GeminiClient(
        model_name=model_name,
        temperature=0.7,  # Higher temperature for creative adversarial generation
        system_prompt=_SYSTEM_PROMPT,
        api_key=api_key,
    )
    raw = client.call(
        f"Generate adversarial search queries to find counterevidence for:\n\n{sub_hyp_text}"
    )
    if raw and "queries" in raw:
        return [str(q) for q in raw["queries"][:3]]
    # Fallback
    return [f"criticism OR replication failure OR counterevidence: {sub_hyp_text[:150]}"]


async def _tavily_search_adversarial(
    queries: list[str],
    api_key: str,
    paper_id: int,
    db_conn,
) -> list[EvidenceItem]:
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — skipping Agent 5 Tavily search")
        return []

    tasks = []
    for q in queries:
        from core.database import atomic_increment_tavily_counter
        allowed = atomic_increment_tavily_counter(db_conn, paper_id, TAVILY_QUERY_CAP)
        if not allowed:
            logger.info("Tavily cap reached — skipping adversarial query: %s", q[:60])
            continue
        tasks.append(_tavily_single_adversarial(q, api_key))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[EvidenceItem] = []
    for r in results:
        if isinstance(r, list):
            items.extend(r)
        elif isinstance(r, Exception):
            logger.warning("Agent 5 Tavily error: %s", r)
    return items


async def _tavily_single_adversarial(query: str, api_key: str) -> list[EvidenceItem]:
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
            agent_source="agent5",
        ))
    return items
