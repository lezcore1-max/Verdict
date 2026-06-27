"""
orchestration/paper_runner.py — Top-level pipeline orchestrator.

Execution flow:
  1. Pre-ingest all PDFs into ChromaDB
  2. Extract + store paper text in SQLite
  3. Run Agent 1 → insert claims → resolve dependency pairs to real IDs
  4. For each claim: ProcessPoolExecutor with 8-minute timeout
  5. Build dependency DAG → propagate scores
  6. generate_paper_summary → write to paper_summaries table
"""
import logging
import sqlite3
import os
import json
import re
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

import numpy as np

from core.config import (
    GEMINI_MODEL, DECOMPOSER_MODEL, SYNTHESIZER_MODEL,
    CHROMA_DIR, TAVILY_API_KEY, EMBED_MODEL
)
from core import database as db
from core.pdf_parser import extract_text, chunk_text
from core.pipeline_state import mark_timed_out

logger = logging.getLogger(__name__)

_CLAIM_TIMEOUT_SECONDS = 8 * 60  # 8 minutes

# Reproducibility flag keywords (case-insensitive substring match)
_REPRO_KEYWORDS = [
    "proprietary",
    "internal dataset",
    "not publicly available",
    "single institution",
    "upon request",
    "our dataset",
]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_paper(
    pdf_path: str,
    db_path: str,
    model_name: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
    tavily_key: Optional[str] = None,
    chroma_dir: str = CHROMA_DIR,
    status_callback=None,   # optional callable(str) for progress reporting
) -> int:
    """
    Run the full VERDICT pipeline for a PDF.

    Args:
        pdf_path:         Absolute path to the PDF file.
        db_path:          Path to the SQLite database file.
        model_name:       Gemini model name.
        api_key:          Gemini API key override.
        tavily_key:       Tavily API key override.
        chroma_dir:       ChromaDB persistence directory.
        status_callback:  Optional callable for UI status updates.

    Returns:
        paper_id (int) from SQLite.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)

    def status(msg: str) -> None:
        logger.info(msg)
        if status_callback:
            try:
                status_callback(msg)
            except Exception:
                pass

    # ── Step 1: Register paper ───────────────────────────────────────────────
    paper_id = db.upsert_paper(conn, pdf_path)
    status(f"📄 Paper registered (id={paper_id})")

    # ── Step 2: Extract text ─────────────────────────────────────────────────
    status("🔍 Extracting text from PDF...")
    full_text = extract_text(pdf_path)
    if not full_text.strip():
        logger.error("PDF text extraction returned empty string for %s", pdf_path)
        conn.close()
        return paper_id
    db.set_paper_text(conn, paper_id, full_text)
    status(f"✅ Text extracted ({len(full_text):,} chars)")

    # ── Step 3: Pre-ingest into ChromaDB ─────────────────────────────────────
    if not db.is_paper_ingested(conn, paper_id):
        status("📚 Ingesting paper into ChromaDB RAG store...")
        try:
            from core.rag import get_chroma_collection, ingest_chunks
            collection = get_chroma_collection(chroma_dir)
            chunks = chunk_text(full_text, chunk_tokens=512, overlap_tokens=64)
            ingest_chunks(collection, paper_id, chunks)
            db.mark_paper_ingested(conn, paper_id)
            status(f"✅ Ingested {len(chunks)} chunks into ChromaDB")
        except Exception as exc:
            logger.warning("ChromaDB ingestion failed: %s", exc)
    else:
        status("✅ Paper already ingested in ChromaDB")

    # ── Step 4: Run Agent 1 ───────────────────────────────────────────────────
    status("🤖 Agent 1: Extracting falsifiable claims...")
    import agents.agent1_claim_extractor as a1
    claim_output = a1.run(full_text, model_name=model_name, api_key=api_key)

    if claim_output is None or not claim_output.claims:
        status("⚠️ Agent 1 returned no claims")
        conn.close()
        return paper_id

    status(f"✅ Extracted {len(claim_output.claims)} claims")

    # ── Step 5: Insert claims + resolve dependency pairs ──────────────────────
    inserted_claims: list[dict] = []  # {"position": int, "db_id": int}
    for claim in claim_output.claims:
        claim_db_id = db.insert_claim(
            conn,
            paper_id=paper_id,
            text=claim.text,
            claim_type=claim.type,
            epistemic_weight=claim.epistemic_weight,
            section=claim.section,
            position=claim.position,
        )
        inserted_claims.append({"position": claim.position, "db_id": claim_db_id})

    # Resolve positional dependency pairs → real SQLite claim IDs
    position_to_id = {c["position"]: c["db_id"] for c in inserted_claims}
    for (from_pos, to_pos) in claim_output.dependency_pairs:
        if from_pos in position_to_id and to_pos in position_to_id:
            db.insert_dependency_edge(conn, position_to_id[from_pos], position_to_id[to_pos])

    status(f"✅ Inserted {len(inserted_claims)} claims + {len(claim_output.dependency_pairs)} dependency edges")

    # ── Step 6: Process each claim ────────────────────────────────────────────
    resume = _get_pending_claim_ids(conn, paper_id)
    status(f"🔄 Processing {len(resume)} claims (ProcessPoolExecutor, 8-min timeout each)...")

    from concurrent.futures import ProcessPoolExecutor

    for claim_db_id in resume:
        status(f"  ⏳ Processing claim id={claim_db_id}...")
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _run_claim_subprocess,
                    claim_db_id,
                    paper_id,
                    db_path,
                    model_name,
                    api_key,
                    tavily_key,
                    chroma_dir,
                )
                future.result(timeout=_CLAIM_TIMEOUT_SECONDS)
            status(f"  ✅ Claim {claim_db_id} done")
        except FuturesTimeoutError:
            logger.warning("Claim %d timed out after %ds", claim_db_id, _CLAIM_TIMEOUT_SECONDS)
            mark_timed_out(conn, claim_db_id)
            status(f"  ⏰ Claim {claim_db_id} timed out")
        except Exception as exc:
            logger.error("Claim %d process error: %s", claim_db_id, exc)
            from core.pipeline_state import mark_failed
            mark_failed(conn, claim_db_id, str(exc))
            status(f"  ❌ Claim {claim_db_id} failed: {exc}")

    # ── Step 7: Dependency DAG + score propagation ────────────────────────────
    status("🔗 Building dependency graph and propagating scores...")
    _propagate_dependency_scores(conn, paper_id)

    # ── Step 8: Paper-level summary ───────────────────────────────────────────
    status("📊 Generating paper-level summary...")
    generate_paper_summary(paper_id, conn)
    status("🎉 Pipeline complete!")

    conn.close()
    return paper_id


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess entry point (must be module-level for pickling on Windows)
# ─────────────────────────────────────────────────────────────────────────────

def _run_claim_subprocess(
    claim_id: int,
    paper_id: int,
    db_path: str,
    model_name: str,
    api_key: Optional[str],
    tavily_key: Optional[str],
    chroma_dir: str,
) -> None:
    """Module-level function so ProcessPoolExecutor can pickle it on Windows."""
    import sys
    import os
    # Ensure the verdict/ root is on sys.path in the subprocess
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from orchestration.claim_runner import run_claim
    run_claim(
        claim_id=claim_id,
        paper_id=paper_id,
        db_path=db_path,
        model_name=model_name,
        api_key=api_key,
        tavily_key=tavily_key,
        chroma_dir=chroma_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dependency propagation
# ─────────────────────────────────────────────────────────────────────────────

def _propagate_dependency_scores(conn: sqlite3.Connection, paper_id: int) -> None:
    """Build DAG and propagate support scores topologically."""
    from verdict_math.dependency_graph import build_dag, propagate_scores, build_dependency_summary

    edges = db.get_dependency_edges(conn, paper_id)
    edge_tuples = [(e["from_claim_id"], e["to_claim_id"]) for e in edges]
    dag = build_dag(edge_tuples)

    # Collect raw support scores
    verdicts = db.get_all_verdicts(conn, paper_id)
    raw_scores = {v["claim_id"]: float(v["support"] or 0.0) for v in verdicts}

    if not dag.nodes():
        return

    # Add any claim IDs not in the DAG
    for claim_id in raw_scores:
        if claim_id not in dag:
            dag.add_node(claim_id)

    adjusted = propagate_scores(dag, raw_scores)

    # Update adjusted_support in verdicts
    for claim_id, adj_support in adjusted.items():
        conn.execute(
            "UPDATE verdicts SET adjusted_support=? WHERE claim_id=?",
            (adj_support, claim_id)
        )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Paper-level summary
# ─────────────────────────────────────────────────────────────────────────────

def generate_paper_summary(paper_id: int, conn: sqlite3.Connection) -> None:
    """
    Deterministic aggregation of all claim verdicts into a paper-level summary.
    No LLM call. Pure Python reading from SQLite.

    Writes to the paper_summaries table.
    """
    verdicts = db.get_all_verdicts(conn, paper_id)
    if not verdicts:
        db.save_paper_summary(
            conn, paper_id,
            overall_support=0.0, overall_contradiction=0.0, overall_uncertainty=1.0,
            top_strong_claims=[], top_weak_claims=[],
            dependency_summary="No claims were processed.",
            formally_supported=[], implied_only=[],
            reproducibility_flag=False,
        )
        return

    # Filter out failed/timed-out claims (those without a verdict row won't appear)
    s_vals = [float(v["support"] or 0) for v in verdicts]
    c_vals = [float(v["contradiction"] or 0) for v in verdicts]
    u_vals = [float(v["uncertainty"] or 1) for v in verdicts]
    adj_vals = [float(v["adjusted_support"] or 0) for v in verdicts]

    overall_support = float(np.mean(s_vals))
    overall_contradiction = float(np.mean(c_vals))
    overall_uncertainty = float(np.mean(u_vals))

    # Top 3 strongest by adjusted_support
    sorted_by_strong = sorted(verdicts, key=lambda v: float(v["adjusted_support"] or 0), reverse=True)
    top_strong_claims = [
        {"claim_id": v["claim_id"], "adjusted_support": float(v["adjusted_support"] or 0)}
        for v in sorted_by_strong[:3]
    ]

    # Top 3 weakest by contradiction + uncertainty
    sorted_by_weak = sorted(
        verdicts,
        key=lambda v: float(v["contradiction"] or 0) + float(v["uncertainty"] or 0),
        reverse=True
    )
    top_weak_claims = [
        {
            "claim_id": v["claim_id"],
            "weakness_score": float(v["contradiction"] or 0) + float(v["uncertainty"] or 0)
        }
        for v in sorted_by_weak[:3]
    ]

    # Formally supported vs implied only
    formally_supported = []
    implied_only = []
    for v in verdicts:
        claim_row = conn.execute("SELECT text FROM claims WHERE id=?", (v["claim_id"],)).fetchone()
        if claim_row is None:
            continue
        text = claim_row["text"]
        adj = float(v["adjusted_support"] or 0)
        conflict = bool(v["conflict_flag"])
        
        # Check if the claim actually underwent a formal statistical test
        formal_count = conn.execute("""
            SELECT COUNT(*) FROM evidence e
            JOIN sub_hypotheses sh ON e.sub_hyp_id = sh.id
            WHERE sh.claim_id = ? AND e.p_value_tag = 'formal'
        """, (v["claim_id"],)).fetchone()[0]

        if adj > 0.6 and not conflict and formal_count > 0:
            formally_supported.append({"claim_id": v["claim_id"], "text": text[:200]})
        else:
            implied_only.append({"claim_id": v["claim_id"], "text": text[:200]})

    # Dependency summary
    edges = db.get_dependency_edges(conn, paper_id)
    n_edges = len(edges)
    adj_scores = {v["claim_id"]: float(v["adjusted_support"] or 0) for v in verdicts}
    weak_preds = [cid for cid, s in adj_scores.items() if s < 0.3]
    dep_summary = (
        f"The paper has {n_edges} inter-claim dependency edge(s). "
        + (
            f"⚠️ {len(weak_preds)} foundational claim(s) have low support (<30%), "
            "which may weaken dependent results."
            if weak_preds
            else "No critical foundational weaknesses detected."
        )
    )

    # Reproducibility flag — deterministic keyword search
    repro_flag = _check_reproducibility_flag(conn, paper_id)

    db.save_paper_summary(
        conn,
        paper_id=paper_id,
        overall_support=overall_support,
        overall_contradiction=overall_contradiction,
        overall_uncertainty=overall_uncertainty,
        top_strong_claims=top_strong_claims,
        top_weak_claims=top_weak_claims,
        dependency_summary=dep_summary,
        formally_supported=formally_supported,
        implied_only=implied_only,
        reproducibility_flag=repro_flag,
    )
    logger.info("Paper summary saved for paper_id=%d", paper_id)


def _check_reproducibility_flag(conn: sqlite3.Connection, paper_id: int) -> bool:
    """
    Deterministic keyword search across the full raw text of the paper.
    Case-insensitive. Returns True if any reproducibility concern keyword is found.
    """
    full_text = db.get_paper_text(conn, paper_id)
    if not full_text:
        return False
        
    full_text = full_text.lower()
    for kw in _REPRO_KEYWORDS:
        if kw in full_text:
            return True
            
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_pending_claim_ids(conn: sqlite3.Connection, paper_id: int) -> list[int]:
    """Return claim IDs that are not yet done/timed_out/failed."""
    rows = conn.execute(
        """
        SELECT id FROM claims
         WHERE paper_id = ? AND status NOT IN ('done', 'timed_out', 'failed')
         ORDER BY position
        """,
        (paper_id,),
    ).fetchall()
    return [r["id"] for r in rows]
