"""
agents/agent2_hypothesis_decomposer.py — Decompose a claim into sub-hypotheses.

CRITICAL INVARIANT: This agent NEVER sees the paper text.
It receives ONLY claim_text and claim_type.
This preserves the sequential information property required for SPRT independence.
"""
import logging
from typing import Optional

from core.config import DECOMPOSER_MODEL
from core.gemini_client import GeminiClient
from agents.schemas import HypothesisDecompOutput, SubHypothesis

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a scientific hypothesis decomposer. You receive a single claim from a machine learning research paper and decompose it into 2-3 independent, falsifiable sub-hypotheses.

Rules:
1. Each sub-hypothesis must be a NECESSARY CONDITION of the main claim — if any sub-hypothesis is false, the main claim is false.
2. Each sub-hypothesis must be independently testable with external evidence.
3. Sub-hypotheses must be INDEPENDENT of each other.
4. State the logical_relationship explicitly for each (e.g., "necessary condition", "independent supporting condition").
5. Do NOT reference the original paper, its authors, or specific experimental details. Frame sub-hypotheses as general testable propositions.
6. Output 2 to 3 sub-hypotheses maximum.

You will receive the claim text and its type tag.

Respond ONLY with a valid JSON object. No markdown fences, no explanation, no preamble."""


def run(
    claim_text: str,
    claim_type: str,
    model_name: str = DECOMPOSER_MODEL,
    api_key: Optional[str] = None,
) -> Optional[HypothesisDecompOutput]:
    """
    Decompose a claim into 2-3 independent falsifiable sub-hypotheses.

    Args:
        claim_text:  The bare claim text. NO paper context.
        claim_type:  The claim type tag (e.g., "benchmark_performance").
        model_name:  Gemini model to use (DECOMPOSER_MODEL by default).
        api_key:     Optional override for GEMINI_API_KEY.

    Returns:
        HypothesisDecompOutput or None on extraction failure.

    INVARIANT: This function signature intentionally has NO paper_text parameter.
    """
    # ── Hard invariant enforcement ──────────────────────────────────────────
    # The function signature itself enforces this — no paper_text arg exists.
    # This comment serves as the explicit documentation of the invariant.
    # ────────────────────────────────────────────────────────────────────────

    client = GeminiClient(
        model_name=model_name,
        temperature=0.2,
        system_prompt=_SYSTEM_PROMPT,
        api_key=api_key,
    )

    prompt = (
        f"Claim type: {claim_type}\n\n"
        f"Claim: {claim_text}\n\n"
        "Decompose this claim into 2-3 independent falsifiable sub-hypotheses "
        "that are necessary conditions of the main claim."
    )

    raw = client.call(prompt)
    if raw is None:
        logger.warning("Agent 2: decomposition failed for claim: %s", claim_text[:80])
        return None

    try:
        # Normalise possible model output formats
        if isinstance(raw, list):
            raw = {"sub_hypotheses": raw}
        if "sub_hypotheses" not in raw:
            # Try to find a list in the response
            for key in raw:
                if isinstance(raw[key], list):
                    raw = {"sub_hypotheses": raw[key]}
                    break

        result = HypothesisDecompOutput.model_validate(raw)

        # Assign sequential positions
        for i, sh in enumerate(result.sub_hypotheses):
            sh.position = i

        logger.info("Agent 2: generated %d sub-hypotheses", len(result.sub_hypotheses))
        return result

    except Exception as exc:
        logger.warning("Agent 2: Pydantic validation failed: %s", exc)
        # Graceful fallback: extract from raw list
        try:
            subs_raw = raw.get("sub_hypotheses", []) if isinstance(raw, dict) else []
            subs = []
            for i, s in enumerate(subs_raw[:3]):
                if isinstance(s, dict):
                    text_val = s.get("text") or s.get("hypothesis") or s.get("sub_hypothesis")
                    if text_val:
                        subs.append(SubHypothesis(
                            text=text_val,
                            logical_relationship=s.get("logical_relationship", "necessary condition"),
                            position=i,
                        ))
            if subs:
                return HypothesisDecompOutput(sub_hypotheses=subs)
        except Exception:
            pass
        return None
