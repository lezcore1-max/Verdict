"""
agents/agent1_claim_extractor.py — Extract falsifiable claims from paper text.

Chunking strategy:
  - If paper fits in one context window, single call.
  - Otherwise: 4000-token / 500-token overlap chunks, merge + deduplicate.

Deduplication:
  - Exact string match first (O(1))
  - Then cosine similarity ≥ 0.85 via LocalEmbedder
"""
import logging
from typing import Optional

from core.config import GEMINI_MODEL, CLAIM_DEDUP
from core.gemini_client import GeminiClient
from core.embedder import LocalEmbedder
from core.pdf_parser import chunk_for_agent1, detect_section, epistemic_weight_for_section, hedge_adjustment
from agents.schemas import ClaimExtractOutput, ExtractedClaim

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a scientific claim extraction specialist. Your task is to read machine learning research paper text and extract all falsifiable claims.

A falsifiable claim must:
- Make a specific, testable assertion about empirical results, causal relationships, comparisons, mechanisms, novelty, or generalizability
- Be specific enough that external evidence could confirm or contradict it
- NOT be a vague aspiration, future work statement, or definition

For each claim, determine:
- type: one of benchmark_performance, causal, comparative, mechanistic, novelty, generalization
- epistemic_weight: 0.0-1.0 based on section (results/abstract=high, discussion/conclusion=low) and language (demonstrates/proves=high, suggests/may=low)
- section: the paper section where this claim appears
- position: 0-indexed position in your output list

Also identify dependency_pairs: pairs of positional indices (from_position, to_position) where one claim's validity logically depends on another. These are POSITIONAL INDICES into your claims list, NOT database IDs.

Respond ONLY with a valid JSON object. No markdown fences, no explanation, no preamble. Your output must strictly follow this structure:
{
  "claims": [
    {
      "text": "The extracted claim text",
      "type": "comparative",
      "epistemic_weight": 0.9,
      "section": "Abstract",
      "position": 0
    }
  ],
  "dependency_pairs": [
    [0, 1]
  ]
}"""

# Approximate token limit per chunk before we need to split
_SINGLE_CALL_TOKEN_LIMIT = 3500


def run(
    paper_text: str,
    model_name: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
) -> Optional[ClaimExtractOutput]:
    """
    Extract claims from paper text.

    Returns ClaimExtractOutput with merged, deduplicated claims and
    positional dependency pairs, or None on complete failure.
    """
    chunks = chunk_for_agent1(paper_text)

    if len(chunks) == 1:
        return _run_single_chunk(chunks[0], model_name, api_key)

    # Multiple chunks: run per chunk, merge, deduplicate
    all_claims: list[ExtractedClaim] = []
    all_pairs: list[tuple[int, int]] = []
    base_idx = 0
    
    for i, chunk in enumerate(chunks):
        logger.info("Agent 1: processing chunk %d/%d", i + 1, len(chunks))
        result = _run_single_chunk(chunk, model_name, api_key)
        if result:
            all_claims.extend(result.claims)
            for f, t in result.dependency_pairs:
                all_pairs.append((f + base_idx, t + base_idx))
            base_idx += len(result.claims)

    if not all_claims:
        return None

    # Deduplicate merged claims
    embedder = LocalEmbedder()
    deduped = embedder.deduplicate(
        all_claims,
        key_fn=lambda c: c.text,
        threshold=CLAIM_DEDUP,
        exact_key_fn=lambda c: c.text,
    )

    # Map id(claim) and text to its new deduped position
    kept_ids = {id(c): i for i, c in enumerate(deduped)}
    kept_texts = {c.text: i for i, c in enumerate(deduped)}
    
    # Map old global index to new position (for kept claims and exact text duplicates)
    old_to_new = {}
    for old_i, claim in enumerate(all_claims):
        if id(claim) in kept_ids:
            old_to_new[old_i] = kept_ids[id(claim)]
        elif claim.text in kept_texts:
            old_to_new[old_i] = kept_texts[claim.text]

    # Filter and remap edges
    valid_pairs = []
    for f, t in all_pairs:
        if f in old_to_new and t in old_to_new:
            valid_pairs.append((old_to_new[f], old_to_new[t]))

    # Re-assign sequential positions after dedup
    for i, claim in enumerate(deduped):
        claim.position = i

    return ClaimExtractOutput(claims=deduped, dependency_pairs=valid_pairs)


def _run_single_chunk(
    text: str,
    model_name: str,
    api_key: Optional[str],
) -> Optional[ClaimExtractOutput]:
    client = GeminiClient(
        model_name=model_name,
        temperature=0.2,
        system_prompt=_SYSTEM_PROMPT,
        api_key=api_key,
    )
    prompt = f"Extract all falsifiable claims from the following research paper text:\n\n{text}"
    raw = client.call(prompt)
    if raw is None:
        logger.warning("Agent 1: extraction failed (None returned)")
        return None

    try:
        # Normalise: the model may return {"claims": [...]} or a list directly
        if isinstance(raw, list):
            raw = {"claims": raw, "dependency_pairs": []}
        elif "items" in raw and "claims" not in raw:
            raw["claims"] = raw["items"]
            
        if "claims" not in raw:
            raw["claims"] = []
        if "dependency_pairs" not in raw:
            raw["dependency_pairs"] = []

        result = ClaimExtractOutput.model_validate(raw)
        logger.info("Agent 1: extracted %d claims", len(result.claims))
        return result
    except Exception as exc:
        logger.warning("Agent 1: Pydantic validation failed: %s", exc)
        # Attempt graceful recovery: build minimal valid output
        try:
            claims_raw = raw.get("claims", [])
            claims = []
            for i, c in enumerate(claims_raw):
                if isinstance(c, dict) and "text" in c:
                    section = c.get("section", detect_section(c["text"]))
                    base_weight = epistemic_weight_for_section(section)
                    hedge = hedge_adjustment(c.get("text", ""))
                    claims.append(ExtractedClaim(
                        text=c["text"],
                        type=c.get("type", "comparative"),
                        epistemic_weight=round(base_weight * hedge, 3),
                        section=section,
                        position=i,
                    ))
            return ClaimExtractOutput(claims=claims, dependency_pairs=[])
        except Exception:
            return None
