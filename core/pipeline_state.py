"""
core/pipeline_state.py — Crash recovery helpers.

The orchestrator writes a pipeline_state row before and after each stage.
On restart, get_resume_state reads these rows to skip already-completed work.
"""
import logging
import sqlite3
from typing import Optional

from core import database as db

logger = logging.getLogger(__name__)


def get_resume_state(conn: sqlite3.Connection, paper_id: int) -> dict:
    """
    Determine which claims still need processing for this paper.

    Returns:
        {
          "pending_claim_ids": list[int],   # claims not yet done
          "done_claim_ids":    list[int],   # already completed
        }
    """
    claims = db.get_claims_for_paper(conn, paper_id)
    pending = []
    done = []
    for claim in claims:
        if db.is_claim_done(conn, claim["id"]):
            done.append(claim["id"])
        else:
            pending.append(claim["id"])

    logger.info(
        "Resume state for paper %d: %d pending, %d done",
        paper_id, len(pending), len(done)
    )
    return {"pending_claim_ids": pending, "done_claim_ids": done}


def mark_timed_out(conn: sqlite3.Connection, claim_id: int) -> None:
    """Mark a claim as timed out and log the state transition."""
    db.update_claim_status(conn, claim_id, "timed_out")
    db.update_pipeline_state(conn, claim_id, "timeout", "timed_out")
    logger.warning("Claim %d marked as TIMED_OUT", claim_id)


def mark_failed(conn: sqlite3.Connection, claim_id: int, reason: str = "") -> None:
    """Mark a claim as failed."""
    db.update_claim_status(conn, claim_id, "failed")
    db.update_pipeline_state(conn, claim_id, "failed", f"failed: {reason}")
    logger.error("Claim %d marked as FAILED: %s", claim_id, reason)
