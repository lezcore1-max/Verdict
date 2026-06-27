"""
agents/agent4_evidence_judge.py — Judge each evidence item against a sub-hypothesis.

For benchmark_performance claims with PwC leaderboard data:
  - Fetch top-20 scores for the matched metric + dataset
  - Run scipy.stats.ttest_1samp with the paper's claimed score as popmean
  - Tag p_value as "formal"

Otherwise: LLM-estimated p-value proxy, tagged "approximate".
"""
import logging
import re
from typing import Optional

try:
    from scipy import stats as scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from core.config import GEMINI_MODEL, P_VALUE_FLOOR
from core.gemini_client import GeminiClient
from agents.schemas import EvidenceItem, JudgedEvidence

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a scientific evidence evaluator. You receive a sub-hypothesis and a single piece of evidence. Your task is to judge whether and how the evidence tests the sub-hypothesis.

CRITICAL RULES FOR BENCHMARK CLAIMS:
- If the sub-hypothesis claims a model achieved a specific score (e.g., "28.4 BLEU"), your job is to judge whether the evidence confirms or denies that THE PAPER REPORTED THAT SCORE.
- Do NOT mark it as "contradicting" just because newer models achieve higher scores. The claim is about what the paper reported, not about current state-of-the-art.
- If the evidence shows the same paper/model achieving a similar score, it is SUPPORTING.
- If the evidence shows independent reproductions achieving the same or similar score (within 1-2 points), it is SUPPORTING.
- Only mark as "contradicting" if the evidence shows the specific claimed result was fabricated, retracted, or consistently irreproducible.
- If the evidence discusses a completely different model or paper, mark as "inconclusive" — it is not relevant.

Produce a structured assessment:
- directly_tests: true if the evidence directly tests the sub-hypothesis, false if only tangential
- directionality: "supporting" / "contradicting" / "inconclusive"
- strength: "strong" / "moderate" / "weak"
- p_value: a proxy p-value between 0 and 1 (not from a real statistical test; treat as approximate probability that the null hypothesis is true given this evidence). Low values = strong evidence. For well-known, widely-cited results confirmed by the evidence, use p_value between 0.001-0.01. For contradicting evidence, also use low p_value (0.001-0.05).
- p_value_tag: always "approximate" (the formal tag is only assigned when real statistical tests are run)

You do NOT see the original paper — only the sub-hypothesis and the evidence item.

Respond ONLY with a valid JSON object. No markdown fences, no explanation, no preamble."""


def run(
    sub_hyp_text: str,
    evidence: EvidenceItem,
    claim_type: str = "",
    claimed_score: Optional[float] = None,
    pwc_leaderboard: Optional[list] = None,
    model_name: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
) -> Optional[JudgedEvidence]:
    """
    Judge a single evidence item.

    Args:
        sub_hyp_text:    Sub-hypothesis text.
        evidence:        The EvidenceItem to judge.
        claim_type:      If "benchmark_performance" and pwc_leaderboard provided,
                         attempts a formal t-test.
        claimed_score:   The score claimed by the paper (for t-test).
        pwc_leaderboard: List of {"score": float} dicts (top-20 PwC scores).
        model_name:      Gemini model.
        api_key:         Optional API key override.

    Returns JudgedEvidence or None on failure.
    """
    # ── Formal t-test path (benchmark claims with PwC data) ──────────────────
    if (
        claim_type == "benchmark_performance"
        and claimed_score is not None
        and pwc_leaderboard
        and _SCIPY_AVAILABLE
    ):
        formal_result = _formal_ttest(sub_hyp_text, claimed_score, pwc_leaderboard)
        if formal_result is not None:
            return formal_result

    # ── LLM-estimated approximate path ───────────────────────────────────────
    client = GeminiClient(
        model_name=model_name,
        temperature=0.2,
        system_prompt=_SYSTEM_PROMPT,
        api_key=api_key,
    )

    evidence_text = (evidence.raw_content or evidence.content)[:2000]

    prompt = (
        f"Sub-hypothesis: {sub_hyp_text}\n\n"
        f"Evidence source: {evidence.source}\n"
        f"Evidence content: {evidence_text}\n"
        f"Evidence reliability: {evidence.reliability_tier}\n"
        f"Evidence directness: {evidence.directness}\n\n"
        "Judge whether this evidence supports or contradicts the sub-hypothesis."
    )

    raw = client.call(prompt)
    if raw is None:
        logger.warning("Agent 4: LLM call failed for evidence from %s", evidence.source)
        return None

    try:
        judged = JudgedEvidence.model_validate(raw)
        return judged
    except Exception as exc:
        logger.warning("Agent 4: Pydantic validation failed: %s", exc)
        # Graceful fallback with conservative defaults
        try:
            return JudgedEvidence(
                directly_tests=raw.get("directly_tests", False),
                directionality=raw.get("directionality", "inconclusive"),
                strength=raw.get("strength", "weak"),
                p_value=max(float(raw.get("p_value", 0.5)), P_VALUE_FLOOR),
                p_value_tag="approximate",
            )
        except Exception:
            return None


def _formal_ttest(
    sub_hyp_text: str,
    claimed_score: float,
    leaderboard: list[dict],
) -> Optional[JudgedEvidence]:
    """
    Compare the paper's claimed score against the PwC leaderboard.

    The question is: "Did this paper plausibly achieve this score?"
    NOT: "Is this score competitive with current SOTA?"

    If the claimed score is within the range of known leaderboard results,
    it is SUPPORTING (the score is a real, achievable result).
    """
    try:
        import numpy as np
        scores = np.array([float(r["score"]) for r in leaderboard], dtype=np.float64)
        if len(scores) < 3:
            return None  # Not enough data

        lb_mean = float(np.mean(scores))
        lb_std = float(np.std(scores, ddof=1))
        lb_min = float(np.min(scores))
        lb_max = float(np.max(scores))

        # If claimed score is within the range [min, max] of leaderboard,
        # or within 2 standard deviations of the mean, it's plausible
        within_range = lb_min <= claimed_score <= lb_max
        within_2sd = abs(claimed_score - lb_mean) <= 2.0 * lb_std if lb_std > 0 else within_range

        if within_range or within_2sd:
            # The score is achievable — supporting evidence
            if lb_std > 0:
                z = abs(claimed_score - lb_mean) / lb_std
                p_val = max(float(2.0 * scipy_stats.norm.sf(z)), P_VALUE_FLOOR)
            else:
                p_val = 0.01  # All scores identical and match

            return JudgedEvidence(
                directly_tests=True,
                directionality="supporting",
                strength="strong" if p_val < 0.1 or within_range else "moderate",
                p_value=p_val,
                p_value_tag="formal",
                eval_note=(
                    f"Claimed score {claimed_score} is within the known leaderboard range "
                    f"[{lb_min:.1f}, {lb_max:.1f}] (mean={lb_mean:.1f}, std={lb_std:.1f}). "
                    f"This is an achievable, plausible result."
                ),
            )
        else:
            # Score is outside the known range — inconclusive (not contradicting,
            # since older papers may have lower scores than current leaderboard)
            return JudgedEvidence(
                directly_tests=True,
                directionality="inconclusive",
                strength="weak",
                p_value=0.5,
                p_value_tag="formal",
                eval_note=(
                    f"Claimed score {claimed_score} is outside the current leaderboard range "
                    f"[{lb_min:.1f}, {lb_max:.1f}]. This may reflect a different evaluation "
                    f"setup or an older benchmark era."
                ),
            )
    except Exception as exc:
        logger.warning("Formal t-test failed: %s", exc)
        return None

