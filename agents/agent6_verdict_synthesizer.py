"""
agents/agent6_verdict_synthesizer.py — Synthesise final per-claim verdict.

Receives all sub-hypothesis verdicts, SPRT results, DS masses, conflict flags,
and inter-agent disagreement scores. Produces a ClaimVerdict with belief triplet
and plain-language explanation.
"""
import logging
from typing import Optional

from core.config import SYNTHESIZER_MODEL
from core.gemini_client import GeminiClient
from agents.schemas import ClaimVerdict

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a scientific verdict synthesizer. You receive the results of a multi-agent hypothesis testing pipeline for a single claim from a machine learning research paper.

Your inputs include:
- The original claim text
- Sub-hypotheses and their individual verdicts
- SPRT (Sequential Probability Ratio Test) results per sub-hypothesis
- Dempster-Shafer belief masses (support, contradiction, uncertainty) per sub-hypothesis
- Conflict flags (HIGH CONFLICT if triggered)
- Inter-agent disagreement score

Synthesise a final verdict as a belief triplet {support, contradiction, uncertainty} that sums to 1.0, plus a plain-language explanation.

The plain-language explanation MUST cover:
1. What is SUPPORTED by the evidence
2. What is CONTRADICTED by the evidence
3. What remains UNCERTAIN and why
4. Any scope qualifications (e.g., "only valid on specific benchmarks", "results may not generalise")
5. If conflict_flag is true for any sub-hypothesis, explicitly note "HIGH CONFLICT: verdict reliability is limited"

Return a JSON object:
{
  "support": float,          // 0.0-1.0
  "contradiction": float,    // 0.0-1.0
  "uncertainty": float,      // 0.0-1.0
  "conflict_flag": bool,
  "plain_language": "string"
}

The three float values should sum to approximately 1.0.

Respond ONLY with a valid JSON object. No markdown fences, no explanation, no preamble."""


def run(
    claim_text: str,
    sub_hyp_verdicts: list[dict],
    sprt_results: list[dict],
    ds_masses: list[dict],
    conflict_flags: list[bool],
    disagreement_score: float,
    model_name: str = SYNTHESIZER_MODEL,
    api_key: Optional[str] = None,
) -> Optional[ClaimVerdict]:
    """
    Synthesise final claim verdict.

    Args:
        claim_text:          Original claim text.
        sub_hyp_verdicts:    List of per-sub-hypothesis verdict dicts.
        sprt_results:        List of per-sub-hypothesis SPRT result dicts.
        ds_masses:           List of per-sub-hypothesis DS mass dicts.
        conflict_flags:      List of bool conflict flags per sub-hypothesis.
        disagreement_score:  Float — variance between Agent 4 and Agent 5 p-values.
        model_name:          Gemini model (SYNTHESIZER_MODEL by default).
        api_key:             Optional API key override.

    Returns ClaimVerdict or None on failure.
    """
    client = GeminiClient(
        model_name=model_name,
        temperature=0.2,
        system_prompt=_SYSTEM_PROMPT,
        api_key=api_key,
    )

    # Build structured context for the synthesizer
    sub_hyp_context = []
    for i, (shv, sprt, dsm, cf) in enumerate(
        zip(sub_hyp_verdicts, sprt_results, ds_masses, conflict_flags)
    ):
        sub_hyp_context.append({
            "index": i,
            "sub_hypothesis": shv.get("text", ""),
            "sprt_decision": sprt.get("decision", "UNDECIDED"),
            "sprt_product": sprt.get("product", 1.0),
            "ds_support": dsm.get("support", 0.0),
            "ds_contradiction": dsm.get("contradiction", 0.0),
            "ds_uncertainty": dsm.get("uncertainty", 1.0),
            "conflict_flag": cf,
        })

    prompt = (
        f"Claim: {claim_text}\n\n"
        f"Inter-agent disagreement score: {disagreement_score:.4f}\n\n"
        f"Sub-hypothesis verdicts:\n{_format_sub_verdicts(sub_hyp_context)}\n\n"
        "Synthesise the final verdict for this claim."
    )

    raw = client.call(prompt)
    if raw is None:
        logger.warning("Agent 6: synthesis failed — using fallback aggregation")
        return _fallback_verdict(ds_masses, conflict_flags)

    try:
        verdict = ClaimVerdict.model_validate(raw)
        # Propagate conflict flag from any sub-hypothesis
        if any(conflict_flags):
            verdict.conflict_flag = True
        return verdict
    except Exception as exc:
        logger.warning("Agent 6: Pydantic validation failed: %s", exc)
        return _fallback_verdict(ds_masses, conflict_flags)


def _format_sub_verdicts(sub_hyp_context: list[dict]) -> str:
    lines = []
    for s in sub_hyp_context:
        cf = "⚠️ HIGH CONFLICT" if s["conflict_flag"] else "OK"
        lines.append(
            f"  [{s['index']}] {s['sub_hypothesis'][:100]}\n"
            f"      SPRT: {s['sprt_decision']} (product={s['sprt_product']:.3f})\n"
            f"      DS: support={s['ds_support']:.3f} "
            f"contradiction={s['ds_contradiction']:.3f} "
            f"uncertainty={s['ds_uncertainty']:.3f} [{cf}]"
        )
    return "\n".join(lines)


def _fallback_verdict(
    ds_masses: list[dict],
    conflict_flags: list[bool],
) -> ClaimVerdict:
    """
    Deterministic fallback when Agent 6 LLM call fails.
    Averages DS masses across sub-hypotheses.
    """
    import numpy as np
    if not ds_masses:
        return ClaimVerdict(
            support=0.0,
            contradiction=0.0,
            uncertainty=1.0,
            conflict_flag=any(conflict_flags),
            plain_language=(
                "Verdict synthesis failed. Insufficient evidence to draw conclusions."
            ),
        )

    s_vals = [d.get("support", 0.0) for d in ds_masses]
    c_vals = [d.get("contradiction", 0.0) for d in ds_masses]
    u_vals = [d.get("uncertainty", 1.0) for d in ds_masses]

    avg_s = float(np.mean(s_vals))
    avg_c = float(np.mean(c_vals))
    avg_u = float(np.mean(u_vals))

    # Normalise
    total = avg_s + avg_c + avg_u
    if total > 0:
        avg_s /= total
        avg_c /= total
        avg_u /= total

    return ClaimVerdict(
        support=avg_s,
        contradiction=avg_c,
        uncertainty=avg_u,
        conflict_flag=any(conflict_flags),
        plain_language=(
            f"Automated aggregation (LLM synthesis unavailable). "
            f"Average support={avg_s:.2%}, contradiction={avg_c:.2%}, "
            f"uncertainty={avg_u:.2%}."
            + (" ⚠️ HIGH CONFLICT detected in one or more sub-hypotheses." if any(conflict_flags) else "")
        ),
    )
