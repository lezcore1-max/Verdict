"""
agents/schemas.py — Pydantic v2 models for all agent inputs and outputs.

Key invariants enforced here:
  - dependency_pairs contain positional ints (not DB IDs)
  - p_value_tag is always present on JudgedEvidence
  - sub_hypotheses list is bounded to 2–3 items
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 — Claim Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedClaim(BaseModel):
    text: str = Field(..., description="The falsifiable claim text")
    type: Literal[
        "benchmark_performance",
        "causal",
        "comparative",
        "mechanistic",
        "novelty",
        "generalization",
    ]
    epistemic_weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="0.0–1.0; higher means stronger evidential basis"
    )
    section: str = Field(..., description="Paper section this claim appears in")
    position: int = Field(..., ge=0, description="0-indexed position in output list")


class ClaimExtractOutput(BaseModel):
    claims: list[ExtractedClaim]
    dependency_pairs: list[tuple[int, int]] = Field(
        default_factory=list,
        description=(
            "Positional index pairs (from_position, to_position). "
            "These are indices into the claims list above, NOT database IDs."
        ),
    )

    @model_validator(mode="after")
    def validate_dependency_positions(self) -> "ClaimExtractOutput":
        n = len(self.claims)
        for (a, b) in self.dependency_pairs:
            if not (0 <= a < n and 0 <= b < n):
                raise ValueError(
                    f"Dependency pair ({a},{b}) out of range for {n} claims"
                )
            if a == b:
                raise ValueError(f"Self-loop dependency at position {a}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 — Hypothesis Decomposer
# ─────────────────────────────────────────────────────────────────────────────

class SubHypothesis(BaseModel):
    text: str = Field(..., description="Falsifiable sub-hypothesis text")
    logical_relationship: str = Field(
        ...,
        description=(
            'E.g. "necessary condition", "sufficient condition", '
            '"independent supporting condition"'
        ),
    )
    position: int = Field(..., ge=0)


class HypothesisDecompOutput(BaseModel):
    sub_hypotheses: list[SubHypothesis] = Field(
        ..., min_length=1, max_length=3,
        description="2–3 independent falsifiable sub-hypotheses"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 & 5 — Evidence (raw, unjudged)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    source: str = Field(..., description="URL, DOI, arxiv ID, or description")
    content: str = Field(..., description="Relevant excerpt or summary")
    reliability_tier: Literal["peer_reviewed", "preprint", "blog"]
    directness: Literal["direct_test", "partial_test", "tangential"]
    raw_content: str = Field(..., description="Full raw content before summarisation")
    agent_source: Literal["agent3", "agent5"]

    # Optional dedup identifiers
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None


class EvidenceOutput(BaseModel):
    evidence: list[EvidenceItem]


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 — Evidence Judge
# ─────────────────────────────────────────────────────────────────────────────

class JudgedEvidence(BaseModel):
    directly_tests: bool
    directionality: Literal["supporting", "contradicting", "inconclusive"]
    strength: Literal["strong", "moderate", "weak"]
    p_value: float = Field(
        default=0.5,
        description=(
            "Proxy p-value. Low value = strong evidence against the null. "
            "Always labelled by p_value_tag."
        ),
    )
    p_value_tag: Literal["formal", "approximate"] = Field(
        default="approximate",
        description=(
            '"formal" = from scipy.stats.ttest_1samp on PwC leaderboard data. '
            '"approximate" = LLM-estimated proxy.'
        ),
    )
    eval_note: Optional[str] = Field(
        default=None,
        description="Optional explanatory note regarding the statistical inference.",
    )

    @model_validator(mode="after")
    def clamp_p_value(self) -> "JudgedEvidence":
        """Enforce p_value in [1e-6, 1.0] at schema level."""
        self.p_value = max(min(self.p_value, 1.0), 1e-6)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 — Verdict Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

class ClaimVerdict(BaseModel):
    support: float = Field(default=0.0, description="Support mass (normalised to [0,1])")
    contradiction: float = Field(default=0.0, description="Contradiction mass (normalised to [0,1])")
    uncertainty: float = Field(default=1.0, description="Uncertainty mass (normalised to [0,1])")
    conflict_flag: bool = False
    plain_language: str = Field(
        default="",
        description=(
            "Plain language explanation covering: what is supported, "
            "what is contradicted, what remains uncertain, scope qualifications."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalise_triplet(cls, data):
        """
        Normalise support/contradiction/uncertainty to sum to 1 BEFORE
        any field-level validation.  Allows callers to pass unnormalised
        values (e.g., support=3, contradiction=1) which are scaled correctly.
        """
        if isinstance(data, dict):
            s = float(data.get("support", 0.0))
            c = float(data.get("contradiction", 0.0))
            u = float(data.get("uncertainty", 1.0))
            total = s + c + u
            if total > 0:
                data["support"] = s / total
                data["contradiction"] = c / total
                data["uncertainty"] = u / total
            else:
                data["support"] = 0.0
                data["contradiction"] = 0.0
                data["uncertainty"] = 1.0
        return data
