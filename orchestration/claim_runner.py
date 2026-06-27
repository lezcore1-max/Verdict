"""
orchestration/claim_runner.py — Per-claim pipeline runner.

This module is executed inside a subprocess spawned by ProcessPoolExecutor.
Each subprocess opens its own SQLite connection.
All exceptions are caught; the claim is marked FAILED/TIMED_OUT rather than propagating.
"""
import logging
import sqlite3
from typing import Optional

from core.config import CHROMA_DIR, DECOMPOSER_MODEL, SYNTHESIZER_MODEL
from core import database as db
from core.pipeline_state import mark_failed
from orchestration.graph import compile_graph, VerdictState
from verdict_math.dempster_shafer import combine_all, add_disagreement, compute_disagreement_score
from core.config import P_VALUE_FLOOR

logger = logging.getLogger(__name__)


def run_claim(
    claim_id: int,
    paper_id: int,
    db_path: str,
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
    tavily_key: Optional[str] = None,
    chroma_dir: str = CHROMA_DIR,
) -> None:
    """
    Run the full 6-agent pipeline for a single claim.

    This function is the top-level entry point for a subprocess.
    It opens its own SQLite connection, runs the LangGraph pipeline,
    persists all results, and marks the claim done.

    On any exception: marks claim FAILED in SQLite and returns.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    try:
        # Load claim details
        claim_row = conn.execute(
            "SELECT * FROM claims WHERE id=?", (claim_id,)
        ).fetchone()
        if claim_row is None:
            logger.error("Claim %d not found in database", claim_id)
            return

        claim_text = claim_row["text"]
        claim_type = claim_row["type"]

        db.update_pipeline_state(conn, claim_id, "started", "running")
        db.update_claim_status(conn, claim_id, "running")

        # Build and run the LangGraph pipeline
        graph = compile_graph()
        initial_state: VerdictState = {
            "paper_id": paper_id,
            "claim_id": claim_id,
            "claim_text": claim_text,
            "claim_type": claim_type,
            "model_name": model_name,
            "api_key": api_key,
            "tavily_key": tavily_key,
            "db_path": db_path,
            "chroma_dir": chroma_dir,
            "sub_hypotheses": [],
            "evidence_per_sub": {},
            "judged_per_sub": {},
            "sprt_per_sub": {},
            "ds_per_sub": {},
            "conflict_flags": {},
            "disagreement_scores": {},
            "final_verdict": None,
            "error": None,
        }

        final_state = graph.invoke(initial_state)

        # Persist sub-hypotheses and all intermediate results
        _persist_results(conn, claim_id, final_state)

        # Mark claim done
        db.update_claim_status(conn, claim_id, "done")
        db.update_pipeline_state(conn, claim_id, "completed", "done")
        logger.info("Claim %d completed successfully", claim_id)

    except Exception as exc:
        logger.exception("Claim %d failed: %s", claim_id, exc)
        try:
            mark_failed(conn, claim_id, str(exc))
        except Exception:
            pass
    finally:
        conn.close()


def _persist_results(
    conn: sqlite3.Connection,
    claim_id: int,
    state: VerdictState,
) -> None:
    """Persist all intermediate and final results to SQLite."""
    import json
    import numpy as np

    sub_hyps = state.get("sub_hypotheses", [])

    # Insert sub-hypotheses and save per-sub results
    sub_hyp_id_map: dict[int, int] = {}  # position → db ID
    for sh in sub_hyps:
        pos = sh["position"]
        sh_id = db.insert_sub_hypothesis(
            conn,
            claim_id=claim_id,
            text=sh["text"],
            logical_relationship=sh.get("logical_relationship", ""),
            position=pos,
        )
        sub_hyp_id_map[pos] = sh_id

    # Save evidence, SPRT, DS per sub-hypothesis
    for sh in sub_hyps:
        pos = sh["position"]
        sh_id = sub_hyp_id_map.get(pos)
        if sh_id is None:
            continue

        # Evidence
        judged_list = state.get("judged_per_sub", {}).get(pos, [])
        ev_list = state.get("evidence_per_sub", {}).get(pos, [])
        for i, judged in enumerate(judged_list):
            if judged is None:
                continue
            ev = ev_list[i] if i < len(ev_list) else {}
            db.insert_evidence(
                conn,
                sub_hyp_id=sh_id,
                source=ev.get("source", ""),
                content=ev.get("content", ""),
                reliability_tier=ev.get("reliability_tier", "blog"),
                directness=ev.get("directness", "tangential"),
                p_value=judged.get("p_value"),
                p_value_tag=judged.get("p_value_tag", "approximate"),
                directionality=judged.get("directionality"),
                agent_source=ev.get("agent_source", "agent3"),
                raw_content=ev.get("raw_content", ""),
            )

        # SPRT
        sprt = state.get("sprt_per_sub", {}).get(pos, {})
        if sprt:
            db.save_sprt(
                conn,
                sub_hyp_id=sh_id,
                step_log=sprt.get("step_log", []),
                product=sprt.get("product", 1.0),
                decision=sprt.get("decision", "UNDECIDED"),
            )

        # DS masses
        dsm = state.get("ds_per_sub", {}).get(pos, {})
        if dsm:
            db.save_ds(
                conn,
                sub_hyp_id=sh_id,
                support=dsm.get("support", 0.0),
                contradiction=dsm.get("contradiction", 0.0),
                uncertainty=dsm.get("uncertainty", 1.0),
                conflict_flag=state.get("conflict_flags", {}).get(pos, False),
            )
        db.update_pipeline_state(conn, claim_id, f"sub_{pos}_saved", "done", sub_hyp_id=sh_id)

    # Compute claim-level DS by combining sub-hypothesis DS masses
    ds_values = [state.get("ds_per_sub", {}).get(sh["position"], {}) for sh in sub_hyps]
    masses = [
        np.array([d.get("support", 0), d.get("contradiction", 0), d.get("uncertainty", 1)])
        for d in ds_values if d
    ]
    claim_triplet, claim_conflict = combine_all(masses) if masses else (
        np.array([0.0, 0.0, 1.0]), False
    )
    avg_disagreement = float(
        np.mean([state.get("disagreement_scores", {}).get(sh["position"], 0.0) for sh in sub_hyps])
        if sub_hyps else 0.0
    )
    claim_triplet = add_disagreement(claim_triplet, avg_disagreement)
    any_conflict = claim_conflict or any(
        state.get("conflict_flags", {}).get(sh["position"], False) for sh in sub_hyps
    )

    # Final verdict (may be from Agent 6 or fallback)
    verdict = state.get("final_verdict") or {}
    plain = verdict.get("plain_language", "Pipeline did not produce a verdict.")

    if verdict and verdict.get("support") is not None:
        support = float(verdict["support"])
        contradiction = float(verdict["contradiction"])
        uncertainty = float(verdict["uncertainty"])
    else:
        support = float(claim_triplet[0])
        contradiction = float(claim_triplet[1])
        uncertainty = float(claim_triplet[2])

    db.save_verdict(
        conn,
        claim_id=claim_id,
        support=support,
        contradiction=contradiction,
        uncertainty=uncertainty,
        conflict_flag=any_conflict,
        disagreement_score=avg_disagreement,
        adjusted_support=support,  # raw; dependency propagation updates this
        plain_language=plain,
    )
