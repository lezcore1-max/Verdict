"""
core/database.py — SQLite schema creation and all CRUD helpers.

All connections are opened with check_same_thread=False so that both the
Streamlit main thread (reads) and the orchestrator background thread (writes
via subprocess) can safely use the same database file.  SQLite serialises
concurrent writes via its file-level WAL lock, so cross-process safety is
guaranteed without additional synchronisation.
"""
import sqlite3
import json
import time
import os
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Connection factory
# ─────────────────────────────────────────────────────────────────────────────

def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a WAL-mode connection that is safe for concurrent access."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    NOT NULL UNIQUE,
    full_text       TEXT,
    ingested_at     TEXT,
    status          TEXT    NOT NULL DEFAULT 'pending',
    tavily_count    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS claims (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        INTEGER NOT NULL REFERENCES papers(id),
    text            TEXT    NOT NULL,
    type            TEXT    NOT NULL,
    epistemic_weight REAL   NOT NULL DEFAULT 0.5,
    section         TEXT,
    position        INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'pending',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dependency_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    from_claim_id   INTEGER NOT NULL REFERENCES claims(id),
    to_claim_id     INTEGER NOT NULL REFERENCES claims(id)
);

CREATE TABLE IF NOT EXISTS sub_hypotheses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        INTEGER NOT NULL REFERENCES claims(id),
    text            TEXT    NOT NULL,
    logical_relationship TEXT,
    position        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evidence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sub_hyp_id      INTEGER NOT NULL REFERENCES sub_hypotheses(id),
    source          TEXT,
    content         TEXT,
    reliability_tier TEXT,
    directness      TEXT,
    p_value         REAL,
    p_value_tag     TEXT,
    directionality  TEXT,
    agent_source    TEXT,
    raw_content     TEXT
);

CREATE TABLE IF NOT EXISTS sprt_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sub_hyp_id      INTEGER NOT NULL REFERENCES sub_hypotheses(id),
    step_log        TEXT,
    product         REAL,
    decision        TEXT
);

CREATE TABLE IF NOT EXISTS ds_masses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sub_hyp_id      INTEGER NOT NULL REFERENCES sub_hypotheses(id),
    support         REAL,
    contradiction   REAL,
    uncertainty     REAL,
    conflict_flag   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS verdicts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        INTEGER NOT NULL REFERENCES claims(id),
    support         REAL,
    contradiction   REAL,
    uncertainty     REAL,
    conflict_flag   INTEGER NOT NULL DEFAULT 0,
    disagreement_score REAL,
    adjusted_support   REAL,
    plain_language  TEXT
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        INTEGER,
    agent_name      TEXT,
    input_json      TEXT,
    output_json     TEXT,
    latency_ms      REAL,
    token_count     INTEGER,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pipeline_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id        INTEGER NOT NULL,
    sub_hyp_id      INTEGER,
    stage           TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending',
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_summaries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id            INTEGER NOT NULL REFERENCES papers(id),
    overall_support     REAL,
    overall_contradiction REAL,
    overall_uncertainty REAL,
    top_strong_claims   TEXT,
    top_weak_claims     TEXT,
    dependency_summary  TEXT,
    formally_supported  TEXT,
    implied_only        TEXT,
    reproducibility_flag INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ui_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        INTEGER,
    message         TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not already exist."""
    conn.executescript(_SCHEMA)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Papers
# ─────────────────────────────────────────────────────────────────────────────

def upsert_paper(conn: sqlite3.Connection, file_path: str) -> int:
    """Insert paper or return existing id. Does NOT store full_text yet."""
    cur = conn.execute(
        "INSERT INTO papers(file_path) VALUES(?) ON CONFLICT(file_path) DO NOTHING",
        (file_path,)
    )
    conn.commit()
    if cur.lastrowid:
        return cur.lastrowid
    row = conn.execute("SELECT id FROM papers WHERE file_path=?", (file_path,)).fetchone()
    return row["id"]


def wipe_paper_state(conn: sqlite3.Connection, paper_id: int) -> None:
    """Wipes all pipeline data for a paper so it can be re-run from scratch."""
    conn.execute("DELETE FROM agent_logs WHERE claim_id IN (SELECT id FROM claims WHERE paper_id=?)", (paper_id,))
    conn.execute("DELETE FROM pipeline_state WHERE claim_id IN (SELECT id FROM claims WHERE paper_id=?)", (paper_id,))
    conn.execute("DELETE FROM sub_hypotheses WHERE claim_id IN (SELECT id FROM claims WHERE paper_id=?)", (paper_id,))
    conn.execute("DELETE FROM verdicts WHERE claim_id IN (SELECT id FROM claims WHERE paper_id=?)", (paper_id,))
    conn.execute("DELETE FROM dependency_edges WHERE from_claim_id IN (SELECT id FROM claims WHERE paper_id=?) OR to_claim_id IN (SELECT id FROM claims WHERE paper_id=?)", (paper_id, paper_id))
    conn.execute("DELETE FROM claims WHERE paper_id=?", (paper_id,))
    conn.execute("DELETE FROM paper_summaries WHERE paper_id=?", (paper_id,))
    conn.execute("DELETE FROM ui_logs WHERE paper_id=?", (paper_id,))
    conn.execute("UPDATE papers SET status='pending', ingested_at=NULL WHERE id=?", (paper_id,))
    conn.commit()


def set_paper_text(conn: sqlite3.Connection, paper_id: int, full_text: str) -> None:
    conn.execute(
        "UPDATE papers SET full_text=?, status='text_extracted' WHERE id=?",
        (full_text, paper_id)
    )
    conn.commit()


def mark_paper_ingested(conn: sqlite3.Connection, paper_id: int) -> None:
    conn.execute(
        "UPDATE papers SET ingested_at=datetime('now'), status='ingested' WHERE id=?",
        (paper_id,)
    )
    conn.commit()


def is_paper_ingested(conn: sqlite3.Connection, paper_id: int) -> bool:
    row = conn.execute("SELECT ingested_at FROM papers WHERE id=?", (paper_id,)).fetchone()
    return row is not None and row["ingested_at"] is not None


def get_paper(conn: sqlite3.Connection, paper_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()


# ─────────────────────────────────────────────────────────────────────────────
# Tavily counter (atomic check-and-increment)
# ─────────────────────────────────────────────────────────────────────────────

def atomic_increment_tavily_counter(
    conn: sqlite3.Connection, paper_id: int, cap: int
) -> bool:
    """
    Atomically check tavily_count < cap and increment.
    Returns True if increment succeeded (query allowed), False if cap reached.
    Uses a single UPDATE…WHERE statement — safe across concurrent subprocesses
    because SQLite serialises writes at the WAL level.
    """
    cur = conn.execute(
        """
        UPDATE papers
           SET tavily_count = tavily_count + 1
         WHERE id = ? AND tavily_count < ?
        """,
        (paper_id, cap)
    )
    conn.commit()
    return cur.rowcount > 0


def get_tavily_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT MAX(tavily_count) as c FROM papers")
    return int(cur.fetchone()["c"] or 0)


def insert_ui_log(conn: sqlite3.Connection, paper_id: int, message: str) -> None:
    """Insert a live UI log message."""
    conn.execute(
        "INSERT INTO ui_logs (paper_id, message) VALUES (?, ?)",
        (paper_id, message)
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Claims
# ─────────────────────────────────────────────────────────────────────────────

def insert_claim(
    conn: sqlite3.Connection,
    paper_id: int,
    text: str,
    claim_type: str,
    epistemic_weight: float,
    section: str,
    position: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO claims(paper_id, text, type, epistemic_weight, section, position)
        VALUES(?,?,?,?,?,?)
        """,
        (paper_id, text, claim_type, epistemic_weight, section, position),
    )
    conn.commit()
    return cur.lastrowid


def get_claims_for_paper(conn: sqlite3.Connection, paper_id: int) -> list:
    return conn.execute(
        "SELECT * FROM claims WHERE paper_id=? ORDER BY position", (paper_id,)
    ).fetchall()


def update_claim_status(conn: sqlite3.Connection, claim_id: int, status: str) -> None:
    conn.execute("UPDATE claims SET status=? WHERE id=?", (status, claim_id))
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Dependency edges
# ─────────────────────────────────────────────────────────────────────────────

def insert_dependency_edge(
    conn: sqlite3.Connection, from_claim_id: int, to_claim_id: int
) -> None:
    conn.execute(
        "INSERT INTO dependency_edges(from_claim_id, to_claim_id) VALUES(?,?)",
        (from_claim_id, to_claim_id),
    )
    conn.commit()


def get_dependency_edges(conn: sqlite3.Connection, paper_id: int) -> list:
    return conn.execute(
        """
        SELECT de.from_claim_id, de.to_claim_id
          FROM dependency_edges de
          JOIN claims c ON c.id = de.from_claim_id
         WHERE c.paper_id = ?
        """,
        (paper_id,),
    ).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-hypotheses
# ─────────────────────────────────────────────────────────────────────────────

def insert_sub_hypothesis(
    conn: sqlite3.Connection,
    claim_id: int,
    text: str,
    logical_relationship: str,
    position: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO sub_hypotheses(claim_id, text, logical_relationship, position)
        VALUES(?,?,?,?)
        """,
        (claim_id, text, logical_relationship, position),
    )
    conn.commit()
    return cur.lastrowid


def get_sub_hypotheses(conn: sqlite3.Connection, claim_id: int) -> list:
    return conn.execute(
        "SELECT * FROM sub_hypotheses WHERE claim_id=? ORDER BY position",
        (claim_id,),
    ).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────────────────────────────────────

def insert_evidence(
    conn: sqlite3.Connection,
    sub_hyp_id: int,
    source: str,
    content: str,
    reliability_tier: str,
    directness: str,
    p_value: Optional[float],
    p_value_tag: Optional[str],
    directionality: Optional[str],
    agent_source: str,
    raw_content: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO evidence(sub_hyp_id, source, content, reliability_tier,
                             directness, p_value, p_value_tag, directionality,
                             agent_source, raw_content)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (sub_hyp_id, source, content, reliability_tier, directness,
         p_value, p_value_tag, directionality, agent_source, raw_content),
    )
    conn.commit()
    return cur.lastrowid


def get_evidence(conn: sqlite3.Connection, sub_hyp_id: int) -> list:
    return conn.execute(
        "SELECT * FROM evidence WHERE sub_hyp_id=?", (sub_hyp_id,)
    ).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# SPRT results
# ─────────────────────────────────────────────────────────────────────────────

def save_sprt(
    conn: sqlite3.Connection,
    sub_hyp_id: int,
    step_log: list,
    product: float,
    decision: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO sprt_results(sub_hyp_id, step_log, product, decision) VALUES(?,?,?,?)",
        (sub_hyp_id, json.dumps(step_log), product, decision),
    )
    conn.commit()
    return cur.lastrowid


def get_sprt(conn: sqlite3.Connection, sub_hyp_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM sprt_results WHERE sub_hyp_id=?", (sub_hyp_id,)
    ).fetchone()


# ─────────────────────────────────────────────────────────────────────────────
# Dempster-Shafer masses
# ─────────────────────────────────────────────────────────────────────────────

def save_ds(
    conn: sqlite3.Connection,
    sub_hyp_id: int,
    support: float,
    contradiction: float,
    uncertainty: float,
    conflict_flag: bool,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO ds_masses(sub_hyp_id, support, contradiction, uncertainty, conflict_flag)
        VALUES(?,?,?,?,?)
        """,
        (sub_hyp_id, support, contradiction, uncertainty, int(conflict_flag)),
    )
    conn.commit()
    return cur.lastrowid


def get_ds(conn: sqlite3.Connection, sub_hyp_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM ds_masses WHERE sub_hyp_id=?", (sub_hyp_id,)
    ).fetchone()


# ─────────────────────────────────────────────────────────────────────────────
# Verdicts
# ─────────────────────────────────────────────────────────────────────────────

def save_verdict(
    conn: sqlite3.Connection,
    claim_id: int,
    support: float,
    contradiction: float,
    uncertainty: float,
    conflict_flag: bool,
    disagreement_score: float,
    adjusted_support: float,
    plain_language: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO verdicts(claim_id, support, contradiction, uncertainty,
                             conflict_flag, disagreement_score, adjusted_support,
                             plain_language)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (claim_id, support, contradiction, uncertainty, int(conflict_flag),
         disagreement_score, adjusted_support, plain_language),
    )
    conn.commit()
    return cur.lastrowid


def get_verdict(conn: sqlite3.Connection, claim_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM verdicts WHERE claim_id=?", (claim_id,)
    ).fetchone()


def get_all_verdicts(conn: sqlite3.Connection, paper_id: int) -> list:
    return conn.execute(
        """
        SELECT v.* FROM verdicts v
        JOIN claims c ON c.id = v.claim_id
        WHERE c.paper_id = ?
        """,
        (paper_id,),
    ).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# Agent logs
# ─────────────────────────────────────────────────────────────────────────────

def log_agent_call(
    conn: sqlite3.Connection,
    claim_id: Optional[int],
    agent_name: str,
    input_data: dict,
    output_data: Optional[dict],
    latency_ms: float,
    token_count: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO agent_logs(claim_id, agent_name, input_json, output_json,
                               latency_ms, token_count)
        VALUES(?,?,?,?,?,?)
        """,
        (
            claim_id,
            agent_name,
            json.dumps(input_data),
            json.dumps(output_data) if output_data is not None else None,
            latency_ms,
            token_count,
        ),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state (crash recovery)
# ─────────────────────────────────────────────────────────────────────────────

def update_pipeline_state(
    conn: sqlite3.Connection,
    claim_id: int,
    stage: str,
    status: str,
    sub_hyp_id: Optional[int] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO pipeline_state(claim_id, sub_hyp_id, stage, status, updated_at)
        VALUES(?,?,?,?,datetime('now'))
        """,
        (claim_id, sub_hyp_id, stage, status),
    )
    conn.commit()


def get_resume_point(conn: sqlite3.Connection, claim_id: int) -> Optional[sqlite3.Row]:
    """Return the last recorded pipeline_state row for a claim."""
    return conn.execute(
        """
        SELECT * FROM pipeline_state
         WHERE claim_id = ?
         ORDER BY id DESC LIMIT 1
        """,
        (claim_id,),
    ).fetchone()


def is_claim_done(conn: sqlite3.Connection, claim_id: int) -> bool:
    row = conn.execute(
        "SELECT status FROM claims WHERE id=?", (claim_id,)
    ).fetchone()
    return row is not None and row["status"] in ("done", "timed_out", "failed")


# ─────────────────────────────────────────────────────────────────────────────
# Paper summaries
# ─────────────────────────────────────────────────────────────────────────────

def save_paper_summary(
    conn: sqlite3.Connection,
    paper_id: int,
    overall_support: float,
    overall_contradiction: float,
    overall_uncertainty: float,
    top_strong_claims: list,
    top_weak_claims: list,
    dependency_summary: str,
    formally_supported: list,
    implied_only: list,
    reproducibility_flag: bool,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO paper_summaries(
            paper_id, overall_support, overall_contradiction, overall_uncertainty,
            top_strong_claims, top_weak_claims, dependency_summary,
            formally_supported, implied_only, reproducibility_flag
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            paper_id,
            overall_support,
            overall_contradiction,
            overall_uncertainty,
            json.dumps(top_strong_claims),
            json.dumps(top_weak_claims),
            dependency_summary,
            json.dumps(formally_supported),
            json.dumps(implied_only),
            int(reproducibility_flag),
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_paper_summary(conn: sqlite3.Connection, paper_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM paper_summaries WHERE paper_id=? ORDER BY id DESC LIMIT 1",
        (paper_id,),
    ).fetchone()
