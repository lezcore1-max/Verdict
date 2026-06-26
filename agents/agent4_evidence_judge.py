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

Produce a structured assessment:
- directly_tests: true if the evidence directly tests the sub-hypothesis, false if only tangential
- directionality: "supporting" / "contradicting" / "inconclusive"
- strength: "strong" / "moderate" / "weak"
- p_value: a proxy p-value between 0 and 1 (not from a real statistical test; treat as approximate probability that the null hypothesis is true given this evidence). Low values indicate strong evidence against the null (strong contradiction or confirmation). High values indicate weak evidence.
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

    prompt = (
        f"Sub-hypothesis: {sub_hyp_text}\n\n"
        f"Evidence source: {evidence.source}\n"
        f"Evidence content: {evidence.content}\n"
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
    Run scipy.stats.ttest_1samp against the PwC leaderboard distribution.

    H0: The paper's claimed score is plausible given the leaderboard distribution.
    alternative="less": one-sided test — is the paper's score ABOVE the leaderboard mean?

    Returns JudgedEvidence with p_value_tag="formal", or None on failure.
    """
    try:
        import numpy as np
        scores = np.array([float(r["score"]) for r in leaderboard], dtype=np.float64)
        if len(scores) < 3:
            return None  # Not enough data for a meaningful t-test

        # One-sample t-test: is the claimed_score consistent with leaderboard distribution?
        # A two-sided test will detect if the claimed_score is an outlier in either direction.
        result = scipy_stats.ttest_1samp(scores, popmean=claimed_score, alternative="two-sided")
        p_val = max(float(result.pvalue), P_VALUE_FLOOR)

        if p_val < 0.05:
            # Significant outlier! Too surprising, warranting skepticism.
            directionality = "inconclusive"
            strength = "weak"
            eval_note = f"Claimed score {claimed_score} for '{sub_hyp_text}' is a statistical outlier relative to the leaderboard (p={p_val:.3f}). Warrants further verification."
        else:
            # Plausible claim. Higher p-value = more consistent with known data.
            directionality = "supporting"
            strength = "strong" if p_val > 0.5 else "moderate" if p_val > 0.1 else "weak"
            eval_note = f"Claimed score {claimed_score} for '{sub_hyp_text}' is consistent with the known leaderboard distribution."

        return JudgedEvidence(
            directly_tests=True,
            directionality=directionality,
            strength=strength,
            p_value=p_val,
            p_value_tag="formal",
            eval_note=eval_note,
        )
    except Exception as exc:
        logger.warning("Formal t-test failed: %s", exc)
        return None
