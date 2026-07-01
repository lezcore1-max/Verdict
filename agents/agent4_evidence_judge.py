"""
agents/agent4_evidence_judge.py — Judge each evidence item against a sub-hypothesis.

For benchmark_performance claims with PwC leaderboard data:
  - Fetch top-20 scores for the matched metric + dataset
  - Run scipy.stats.ttest_1samp with the paper's claimed score as popmean
  - Tag p_value as "formal"

If PwC data is not available, we attempt to extract numbers from the evidence text:
  - If numbers are found and the quote is verified, run appropriate scipy tests.
  - Tag p_value as "formal_extracted"

Otherwise (or on extraction failure): 
  - LLM provides a qualitative label which maps to a fixed conservative p-value.
  - Tag p_value as "conservative_label".
"""
import logging
import re
import json
from typing import Optional, Any

try:
    from scipy import stats as scipy_stats
    import numpy as np
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from core.config import GEMINI_MODEL, P_VALUE_FLOOR
from core.gemini_client import GeminiClient
from agents.schemas import EvidenceItem, JudgedEvidence

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a scientific evidence evaluator. You receive a sub-hypothesis and a single piece of purely qualitative evidence (no exact numbers available). Your task is to judge whether and how the evidence tests the sub-hypothesis.

Produce a structured assessment:
- directly_tests: true if the evidence directly tests the sub-hypothesis, false if only tangential. Ask yourself: "Does this source speak to the stated claim directly, or is it addressing a related but different claim?"
- qualitative_label: MUST BE EXACTLY ONE OF: "strong_support", "moderate_support", "weak_or_inconclusive", "moderate_contradiction", "strong_contradiction".
- eval_note: A brief explanation of your reasoning.

You do NOT see the original paper — only the sub-hypothesis and the evidence item.

Respond ONLY with a valid JSON object. No markdown fences, no explanation, no preamble."""


_EXTRACTOR_SYSTEM_PROMPT = """You are a statistical data extractor. Given a sub-hypothesis and an evidence text, find any exact numeric scores, counts, percentages, or metrics that test the sub-hypothesis.

Determine the 'test_type' based on the numbers you find:
- "one_sample_proportion": if the evidence gives a count and total (e.g. "solved 4 out of 20").
- "one_sample_mean": if the evidence gives multiple run scores (e.g., "[85.2, 86.1, 84.9]") compared to a threshold.
- "two_sample_ind": if the evidence gives scores for two different models/groups to compare.
- "single_score_vs_claim": if the evidence gives a single score (e.g. "model X achieved 87.3%") and the claim states a specific threshold.
- "none": if there are no extractable numbers testing the claim.

If you find numbers, you MUST extract the EXACT substring from the evidence text where you found them into 'verbatim_quote'.

Return a JSON object:
{
  "test_type": "...",
  "verbatim_quote": "...",
  "scores_a": [list of floats],
  "scores_b": [list of floats, if two_sample_ind],
  "count": int (if proportion),
  "total": int (if proportion),
  "single_score": float (if single_score_vs_claim)
}
Respond ONLY with valid JSON."""


def _label_to_conservative_pvalue(label: str) -> tuple[float, str, str]:
    """Map qualitative labels to (p_value, directionality, strength)."""
    mapping = {
        "strong_support": (0.05, "supporting", "strong"),
        "moderate_support": (0.20, "supporting", "moderate"),
        "weak_or_inconclusive": (0.50, "inconclusive", "weak"),
        "moderate_contradiction": (0.20, "contradicting", "moderate"),
        "strong_contradiction": (0.05, "contradicting", "strong"),
    }
    return mapping.get(label, (0.50, "inconclusive", "weak"))


def _extract_numbers_and_compute_pvalue(
    sub_hyp_text: str,
    evidence_text: str,
    claimed_score: Optional[float],
    client: GeminiClient
) -> Optional[JudgedEvidence]:
    """
    Attempt to extract numbers and run a scipy statistical test.
    Returns None if extraction fails, no numbers found, quote doesn't match, or scipy fails.
    """
    if not _SCIPY_AVAILABLE:
        return None

    prompt = (
        f"Sub-hypothesis: {sub_hyp_text}\n"
        f"Claimed threshold/score from paper (if any): {claimed_score}\n\n"
        f"Evidence text:\n{evidence_text}\n"
    )

    client.system_prompt = _EXTRACTOR_SYSTEM_PROMPT
    raw = client.call(prompt)
    
    if not raw or raw.get("test_type", "none") == "none":
        return None

    test_type = raw.get("test_type")
    quote = raw.get("verbatim_quote", "")
    
    # Hallucination guard: Verify the quote actually exists in the text
    if not quote or quote not in evidence_text:
        logger.warning("Agent 4 Extractor: Verbatim quote not found in text. Falling back.")
        return None

    try:
        p_val = 0.5
        eval_note = f"Extracted formal stats from quote: '{quote}'"
        
        if test_type == "one_sample_proportion":
            count = int(raw.get("count", 0))
            total = int(raw.get("total", 1))
            # Default to 0.5 expected if claimed_score not provided
            expected = claimed_score if claimed_score is not None else 0.5
            # Ensure expected is a probability
            if expected > 1.0:
                expected = expected / 100.0
            res = scipy_stats.binom_test(count, total, p=expected)
            p_val = float(res)
            
        elif test_type == "one_sample_mean":
            scores = np.array(raw.get("scores_a", []), dtype=np.float64)
            if len(scores) < 2:
                return None
            popmean = claimed_score if claimed_score is not None else 0.0
            res = scipy_stats.ttest_1samp(scores, popmean=popmean)
            p_val = float(res.pvalue)
            
        elif test_type == "two_sample_ind":
            scores_a = np.array(raw.get("scores_a", []), dtype=np.float64)
            scores_b = np.array(raw.get("scores_b", []), dtype=np.float64)
            if len(scores_a) < 2 or len(scores_b) < 2:
                return None
            res = scipy_stats.ttest_ind(scores_a, scores_b, equal_var=False)
            p_val = float(res.pvalue)
            
        elif test_type == "single_score_vs_claim":
            return None
            
        else:
            return None

        p_val = max(p_val, P_VALUE_FLOOR)
        
        directionality = "supporting" if p_val > 0.1 else "contradicting"
        strength = "strong" if p_val < 0.05 or p_val > 0.5 else "moderate"
        if 0.05 <= p_val <= 0.1:
             directionality = "inconclusive"
             strength = "weak"
             
        return JudgedEvidence(
            directly_tests=True,
            directionality=directionality,
            strength=strength,
            p_value=p_val,
            p_value_tag="formal_extracted",
            eval_note=eval_note
        )

    except Exception as exc:
        logger.warning("Agent 4 Extractor: SciPy execution failed: %s", exc)
        return None


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
    Judge a single evidence item using the formal -> extracted -> conservative hierarchy.
    """
    if (
        claim_type == "benchmark_performance"
        and claimed_score is not None
        and pwc_leaderboard
        and _SCIPY_AVAILABLE
    ):
        formal_result = _formal_ttest(sub_hyp_text, claimed_score, pwc_leaderboard)
        if formal_result is not None:
            return formal_result

    evidence_text = (evidence.raw_content or evidence.content)[:3000]

    client = GeminiClient(
        model_name=model_name,
        temperature=0.1,
        system_prompt=_SYSTEM_PROMPT,
        api_key=api_key,
    )

    extracted_result = _extract_numbers_and_compute_pvalue(
        sub_hyp_text, evidence_text, claimed_score, client
    )
    if extracted_result is not None:
        return extracted_result

    client.system_prompt = _SYSTEM_PROMPT
    prompt = (
        f"Sub-hypothesis: {sub_hyp_text}\n\n"
        f"Evidence source: {evidence.source}\n"
        f"Evidence content: {evidence_text}\n"
        f"Evidence reliability: {evidence.reliability_tier}\n"
        f"Evidence directness: {evidence.directness}\n\n"
        "Judge whether this qualitative evidence supports or contradicts the sub-hypothesis."
    )

    raw = client.call(prompt)
    if raw is None:
        logger.warning("Agent 4: Qualitative LLM call failed for evidence from %s", evidence.source)
        return None

    label = raw.get("qualitative_label", "weak_or_inconclusive")
    p_val, directionality, strength = _label_to_conservative_pvalue(label)

    return JudgedEvidence(
        directly_tests=raw.get("directly_tests", True),
        directionality=directionality,
        strength=strength,
        p_value=p_val,
        p_value_tag="conservative_label",
        eval_note=raw.get("eval_note", f"Assigned conservative p-value based on label: {label}")
    )


def _formal_ttest(
    sub_hyp_text: str,
    claimed_score: float,
    leaderboard: list[dict],
) -> Optional[JudgedEvidence]:
    """
    Run scipy.stats.ttest_1samp against the PwC leaderboard distribution.
    """
    try:
        import numpy as np
        scores = np.array([float(r["score"]) for r in leaderboard], dtype=np.float64)
        if len(scores) < 3:
            return None

        result = scipy_stats.ttest_1samp(scores, popmean=claimed_score, alternative="two-sided")
        p_val = max(float(result.pvalue), P_VALUE_FLOOR)

        if p_val < 0.05:
            directionality = "inconclusive"
            strength = "weak"
            eval_note = f"Claimed score {claimed_score} is a statistical outlier (p={p_val:.3f})."
        else:
            directionality = "supporting"
            strength = "strong" if p_val > 0.5 else "moderate" if p_val > 0.1 else "weak"
            eval_note = f"Claimed score {claimed_score} is consistent with known distribution."

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
