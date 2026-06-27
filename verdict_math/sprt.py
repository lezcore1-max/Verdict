"""
math/sprt.py — Wald's Sequential Probability Ratio Test.

All formulas implemented from scratch in NumPy.

Kappa calibrator:  e = 0.5 × p^(-0.5)
SPRT thresholds:   upper = (1-β)/α = 8.0   → FALSIFIED
                   lower = β/(1-α) = 0.222  → INSUFFICIENT_EVIDENCE
"""
import numpy as np
from typing import Any

from core.config import UPPER_THRESHOLD, LOWER_THRESHOLD, P_VALUE_FLOOR


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────

def kappa_e_value(p: float) -> float:
    """
    Kappa calibrator: converts a p-value proxy to an e-value (likelihood ratio).

      e = 0.5 × p^{-0.5}

    p is clamped to P_VALUE_FLOOR (1e-6) before computation to prevent
    division by zero and infinite e-values.

    Derivation intuition:
      - Under H0, p ~ Uniform(0,1), so E[e] = 0.5 × E[p^{-0.5}] = 1.
      - Under H1 (alternative), small p yields large e, driving the product up.
    """
    p_clamped = float(np.maximum(p, P_VALUE_FLOOR))
    return float(0.5 * (p_clamped ** -0.5))


def run_sprt(
    p_values: list[float],
    p_value_tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the sequential product test on a list of p-values.

    Args:
        p_values:    List of p-value proxies (already clamped or not; we clamp here).
        p_value_tags: Parallel list of "formal"/"approximate" labels.
                     If None, all tagged "approximate".

    Returns dict with:
        step_log: list of {"p", "e", "product", "tag"} per step
        product:  final accumulated product
        decision: "FALSIFIED" | "INSUFFICIENT_EVIDENCE" | "UNDECIDED"

    Stopping rule:
        product >= UPPER_THRESHOLD (19.0) → FALSIFIED
        product <= LOWER_THRESHOLD (0.0526) → INSUFFICIENT_EVIDENCE
        Evidence exhausted without crossing either → UNDECIDED
    """
    if p_value_tags is None:
        p_value_tags = ["approximate"] * len(p_values)

    product = np.float64(1.0)   # NumPy float to make intermediate precision explicit
    step_log: list[dict] = []

    for p, tag in zip(p_values, p_value_tags):
        e = kappa_e_value(p)
        product = product * np.float64(e)  # sequential multiplication in NumPy

        step = {
            "p": float(p),
            "e": float(e),
            "product": float(product),
            "tag": tag,
        }
        step_log.append(step)

        # Check stopping boundaries after each multiplication
        if float(product) >= UPPER_THRESHOLD:
            return {
                "step_log": step_log,
                "product": float(product),
                "decision": "FALSIFIED",
            }
        if float(product) <= LOWER_THRESHOLD:
            return {
                "step_log": step_log,
                "product": float(product),
                "decision": "INSUFFICIENT_EVIDENCE",
            }

    # Evidence exhausted without crossing either boundary
    return {
        "step_log": step_log,
        "product": float(product),
        "decision": "UNDECIDED",
    }
