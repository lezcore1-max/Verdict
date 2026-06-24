"""
math/dempster_shafer.py — Dempster-Shafer Theory for belief aggregation.

All formulas implemented from scratch in NumPy.

Frame of discernment: {support (S), contradiction (C), uncertainty (U)}
Belief triplet: m = [s, c, u]  where s + c + u = 1.0

Dempster combination rule (two sources):
  K   = 1 - (s1·c2 + c1·s2)          # conflict mass
  s12 = (s1·s2 + s1·u2 + u1·s2) / K
  c12 = (c1·c2 + c1·u2 + u1·c2) / K
  u12 = (u1·u2) / K

If K < K_CONFLICT (0.1) → raise ConflictError (do not normalise through conflict).
"""
import numpy as np
from typing import Any

from core.config import K_CONFLICT


# ─────────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────────

class ConflictError(Exception):
    """Raised when Dempster combination conflict mass K < K_CONFLICT_THRESHOLD."""


# ─────────────────────────────────────────────────────────────────────────────
# evidence_to_mass
# ─────────────────────────────────────────────────────────────────────────────

def evidence_to_mass(
    p_value: float,
    directionality: str,
    directness: str,
) -> np.ndarray:
    """
    Convert a single judged evidence item into a belief triplet [s, c, u].

    Rules (applied in order):

    1. directness == "tangential"  OR  directionality == "inconclusive":
           → [0.0, 0.0, 1.0]   all mass on uncertainty

    2. directionality == "supporting":
           raw_s = 1.0 - p_value
           raw_c = 0.0
           raw_u = p_value
           (already sums to 1; division is defensive)

    3. directionality == "contradicting":
           raw_s = 0.0
           raw_c = 1.0 - p_value
           raw_u = p_value

    p_value must already be clamped to >= 1e-6 upstream (enforced by the
    JudgedEvidence Pydantic validator).

    Returns:
        np.ndarray of shape (3,) with values in [0,1] summing to 1.
    """
    if directness == "tangential" or directionality == "inconclusive":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if directionality == "supporting":
        s = float(1.0 - p_value)
        c = 0.0
        u = float(p_value)
    else:  # "contradicting"
        s = 0.0
        c = float(1.0 - p_value)
        u = float(p_value)

    triplet = np.array([s, c, u], dtype=np.float64)
    total = triplet.sum()
    # total == 1.0 by construction; division is defensive against float drift
    return triplet / total


# ─────────────────────────────────────────────────────────────────────────────
# combine_two
# ─────────────────────────────────────────────────────────────────────────────

def combine_two(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """
    Apply Dempster's combination rule to two belief triplets.

    Args:
        m1: np.ndarray [s1, c1, u1]
        m2: np.ndarray [s2, c2, u2]

    Returns:
        Combined triplet [s12, c12, u12] normalised by (1 - K).

    Raises:
        ConflictError: if K = 1 - (s1·c2 + c1·s2) < K_CONFLICT_THRESHOLD.
    """
    s1, c1, u1 = m1[0], m1[1], m1[2]
    s2, c2, u2 = m2[0], m2[1], m2[2]

    # Conflict mass: probability mass assigned to the empty set
    K = float(1.0 - (s1 * c2 + c1 * s2))

    if K < K_CONFLICT:
        raise ConflictError(
            f"K={K:.6f} < threshold={K_CONFLICT:.2f} — "
            "sources are in high conflict; combination halted."
        )

    # Dempster normalised combination
    s12 = (s1 * s2 + s1 * u2 + u1 * s2) / K
    c12 = (c1 * c2 + c1 * u2 + u1 * c2) / K
    u12 = (u1 * u2) / K

    result = np.array([s12, c12, u12], dtype=np.float64)
    # Renormalise to correct floating-point drift
    result = result / result.sum()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# combine_all
# ─────────────────────────────────────────────────────────────────────────────

def combine_all(masses: list[np.ndarray]) -> tuple[np.ndarray, bool]:
    """
    Iteratively combine belief triplets left to right.

    Conflict handling:
        - Before each combine_two call, snapshot the current accumulator as
          `last_good`.
        - If ConflictError is raised, restore `last_good` (the triplet BEFORE
          the conflicting evidence item), set conflict_flag = True, and break.
          The conflicting item is discarded.

    Edge cases:
        - Empty list  → ([0.0, 0.0, 1.0], False)   pure uncertainty
        - Single item → (masses[0], False)           no combination needed

    Returns:
        (final_triplet, conflict_flag)
    """
    if len(masses) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), False

    accumulator = masses[0].copy()
    conflict_flag = False

    for mass in masses[1:]:
        last_good = accumulator.copy()   # snapshot BEFORE this combination step
        try:
            accumulator = combine_two(accumulator, mass)
        except ConflictError:
            conflict_flag = True
            accumulator = last_good      # restore pre-conflict state
            break                        # halt; do not process remaining items

    return accumulator, conflict_flag


# ─────────────────────────────────────────────────────────────────────────────
# add_disagreement
# ─────────────────────────────────────────────────────────────────────────────

def add_disagreement(triplet: np.ndarray, variance: float) -> np.ndarray:
    """
    Incorporate inter-agent disagreement into the uncertainty mass.

    The variance between Agent 4 p-value proxies and Agent 5 p-value proxies
    for the same sub-hypothesis is added to the uncertainty component.
    The triplet is then renormalised so all three components still sum to 1.

    Args:
        triplet:  np.ndarray [s, c, u]
        variance: float — variance of p-values across agents

    Returns:
        Renormalised triplet with inflated uncertainty.
    """
    result = triplet.copy()
    result[2] = float(result[2]) + float(variance)   # add to uncertainty
    total = result.sum()
    if total > 0:
        result = result / total
    else:
        result = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# compute_disagreement_score
# ─────────────────────────────────────────────────────────────────────────────

def compute_disagreement_score(
    agent4_p_values: list[float],
    agent5_p_values: list[float],
) -> float:
    """
    Compute variance between Agent 4 and Agent 5 p-value proxies for the
    same sub-hypothesis as a measure of inter-agent disagreement.

    If either list is empty, returns 0.0.

    Implementation (from scratch in NumPy):
        combined = agent4_p_values + agent5_p_values
        variance = Σ (x - mean)² / n
    """
    all_p = np.array(agent4_p_values + agent5_p_values, dtype=np.float64)
    if len(all_p) == 0:
        return 0.0
    mean = all_p.mean()
    variance = float(((all_p - mean) ** 2).mean())   # population variance
    return variance
