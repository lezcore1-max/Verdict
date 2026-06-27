"""
frontend/app.py — VERDICT Streamlit UI.

Threading model (DECIDED):
  - Run button launches run_paper in a daemon threading.Thread.
  - Thread stored in st.session_state["runner_thread"].
  - Polling loop reads SQLite every 2 seconds via st.rerun().
  - Streamlit main thread is READ-ONLY for SQLite.
  - ProcessPoolExecutor subprocesses own all SQLite writes.
"""
import os
import sys
import json
import time
import sqlite3
import threading
import tempfile
import logging
from typing import Optional

import streamlit as st
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Load env before importing core ────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"), override=False)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VERDICT — Hypothesis Testing Framework",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark premium background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a2e 0%, #0a1520 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}
section[data-testid="stSidebar"] * {
    color: #cbd5e0;
}

/* Main header */
.verdict-header {
    text-align: center;
    padding: 2rem 0 1rem;
    background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(159,122,234,0.08));
    border-radius: 16px;
    border: 1px solid rgba(99,179,237,0.15);
    margin-bottom: 1.5rem;
}
.verdict-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed, #9f7aea, #ed64a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin: 0;
}
.verdict-subtitle {
    color: #90cdf4;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* Claim cards */
.claim-card {
    background: linear-gradient(135deg, rgba(26,32,55,0.95), rgba(20,25,45,0.95));
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}
.claim-card:hover {
    border-color: rgba(99,179,237,0.4);
    box-shadow: 0 4px 24px rgba(99,179,237,0.08);
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-done { background: rgba(72,187,120,0.2); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.badge-running { background: rgba(237,137,54,0.2); color: #f6ad55; border: 1px solid rgba(237,137,54,0.3); }
.badge-pending { background: rgba(160,174,192,0.15); color: #a0aec0; border: 1px solid rgba(160,174,192,0.2); }
.badge-failed { background: rgba(245,101,101,0.2); color: #fc8181; border: 1px solid rgba(245,101,101,0.3); }
.badge-timed_out { background: rgba(237,137,54,0.2); color: #f6ad55; border: 1px solid rgba(237,137,54,0.3); }
.badge-conflict { background: rgba(245,101,101,0.3); color: #feb2b2; border: 1px solid rgba(245,101,101,0.5); font-size: 0.75rem; padding: 0.25rem 0.8rem; }
.badge-formal { background: rgba(99,179,237,0.2); color: #90cdf4; border: 1px solid rgba(99,179,237,0.3); }
.badge-approx { background: rgba(159,122,234,0.15); color: #d6bcfa; border: 1px solid rgba(159,122,234,0.2); }

/* SPRT decision badges */
.badge-FALSIFIED { background: rgba(245,101,101,0.2); color: #fc8181; border: 1px solid rgba(245,101,101,0.4); }
.badge-INSUFFICIENT { background: rgba(237,137,54,0.2); color: #f6ad55; border: 1px solid rgba(237,137,54,0.3); }
.badge-UNDECIDED { background: rgba(160,174,192,0.15); color: #a0aec0; border: 1px solid rgba(160,174,192,0.2); }

/* Belief bar */
.belief-bar-container {
    display: flex;
    height: 12px;
    border-radius: 6px;
    overflow: hidden;
    margin: 0.5rem 0;
    background: rgba(0,0,0,0.3);
}
.belief-support { background: linear-gradient(90deg, #48bb78, #38a169); }
.belief-contra { background: linear-gradient(90deg, #fc8181, #e53e3e); }
.belief-uncertain { background: linear-gradient(90deg, #a0aec0, #718096); }

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #90cdf4;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}

/* Metric values */
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
}

/* Summary card */
.summary-card {
    background: linear-gradient(135deg, rgba(15,20,40,0.98), rgba(10,15,30,0.98));
    border: 1px solid rgba(159,122,234,0.25);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 2rem;
}

/* Repro flag */
.repro-flag {
    background: rgba(237,137,54,0.15);
    border: 1px solid rgba(237,137,54,0.4);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #f6ad55;
    font-weight: 500;
}

/* Log viewer */
.log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #718096;
    padding: 0.1rem 0;
}

/* Progress */
.progress-text {
    color: #90cdf4;
    font-size: 0.9rem;
    margin: 0.25rem 0;
}

hr.verdict-hr {
    border: none;
    border-top: 1px solid rgba(99,179,237,0.15);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Database path (per session, stored in session_state)
# ─────────────────────────────────────────────────────────────────────────────

def _get_db_path() -> str:
    if "db_path" not in st.session_state:
        db_dir = os.path.join(tempfile.gettempdir(), "verdict_runs")
        os.makedirs(db_dir, exist_ok=True)
        st.session_state["db_path"] = os.path.join(db_dir, "verdict.db")
    return st.session_state["db_path"]


def _get_db_conn() -> sqlite3.Connection:
    """Read-only connection for the Streamlit main thread."""
    path = _get_db_path()
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="verdict-header">
    <h1 class="verdict-title">⚖️ VERDICT</h1>
    <p class="verdict-subtitle">Multi-Agent Hypothesis Testing Framework for ML Research Papers</p>
    <p style="color:#718096;font-size:0.8rem;margin-top:0.25rem;">
        Condorcet · Wald SPRT · Dempster-Shafer · LangGraph · Gemini
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline launch helpers  (defined BEFORE sidebar so the button can call them)
# ─────────────────────────────────────────────────────────────────────────────

def _launch_pipeline(
    uploaded_file,
    gemini_key: str,
    tavily_key: str,
    model_name: str,
    decomposer_model: str,
    synthesizer_model: str,
    embed_model: str,
    tavily_cap: int,
) -> None:
    """Save PDF, set env vars, launch background thread."""
    # ── Defensive guards (Streamlit can fire disabled buttons on rapid reruns) ──
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF file first.")
        return
    if not gemini_key or not gemini_key.strip():
        st.sidebar.error("Gemini API key is required.")
        return

    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    os.environ["GEMINI_MODEL"] = model_name
    os.environ["DECOMPOSER_MODEL"] = decomposer_model
    os.environ["SYNTHESIZER_MODEL"] = synthesizer_model
    os.environ["EMBED_MODEL"] = embed_model
    os.environ["TAVILY_QUERY_CAP"] = str(tavily_cap)

    db_dir = os.path.dirname(_get_db_path())
    pdf_path = os.path.join(db_dir, uploaded_file.name)
    os.makedirs(db_dir, exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state["pdf_path"] = pdf_path
    if "status_messages" not in st.session_state:
        st.session_state["status_messages"] = []
    else:
        st.session_state["status_messages"].clear()
    
    # Grab the raw list reference to safely append from the background thread
    status_list = st.session_state["status_messages"]

    def _status_cb(msg: str) -> None:
        status_list.append(msg)

    db_path = _get_db_path()

    from core.database import init_db, get_connection, upsert_paper, wipe_paper_state
    init_conn = get_connection(db_path)
    init_db(init_conn)
    init_conn.close()

    reg_conn = get_connection(db_path)
    paper_id = upsert_paper(reg_conn, pdf_path)
    wipe_paper_state(reg_conn, paper_id)
    reg_conn.close()
    st.session_state["paper_id"] = paper_id

    thread = threading.Thread(
        target=_run_paper_thread,
        args=(pdf_path, db_path, model_name, gemini_key, tavily_key, _status_cb, status_list),
        daemon=True,
        name=f"verdict-paper-{paper_id}",
    )
    thread.start()
    st.session_state["runner_thread"] = thread
    st.rerun()


def _run_paper_thread(
    pdf_path: str,
    db_path: str,
    model_name: str,
    api_key: str,
    tavily_key: str,
    status_cb,
    status_list: list,
) -> None:
    """Thread target — calls run_paper and catches all exceptions."""
    try:
        from orchestration.paper_runner import run_paper
        run_paper(
            pdf_path=pdf_path,
            db_path=db_path,
            model_name=model_name,
            api_key=api_key,
            tavily_key=tavily_key,
            status_callback=status_cb,
        )
    except Exception as exc:
        import traceback
        err_msg = traceback.format_exc()
        logging.getLogger(__name__).exception("Pipeline thread failed: %s", exc)
        status_list.append(f"❌ Pipeline error: {exc}\n{err_msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("<hr class='verdict-hr'>", unsafe_allow_html=True)

    # PDF upload
    uploaded_file = st.file_uploader(
        "📄 Upload Research Paper (PDF)",
        type=["pdf"],
        key="pdf_uploader",
    )

    st.markdown("<hr class='verdict-hr'>", unsafe_allow_html=True)
    st.markdown("### 🔑 API Keys")

    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        key="gemini_key_input",
        placeholder="AIza...",
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        key="tavily_key_input",
        placeholder="tvly-...",
    )

    st.markdown("<hr class='verdict-hr'>", unsafe_allow_html=True)
    st.markdown("### 🤖 Model Settings")

    global_model = st.selectbox(
        "Gemini Model (global)",
        options=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-flash-lite", "gemini-3.0-flash", "gemini-3.1-flash-lite"],
        index=0,
        key="global_model",
    )

    advanced_models = st.toggle("Advanced model settings", key="adv_models")
    decomposer_model = global_model
    synthesizer_model = global_model
    if advanced_models:
        decomposer_model = st.selectbox(
            "Agent 2 (Decomposer) model",
            ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash-lite", "gemini-3.0-flash", "gemini-3.1-flash-lite"],
            key="decomp_model",
        )
        synthesizer_model = st.selectbox(
            "Agent 6 (Synthesizer) model",
            ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash-lite", "gemini-3.0-flash", "gemini-3.1-flash-lite"],
            key="synth_model",
        )

    embed_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        key="embed_model_sel",
    )

    tavily_cap = st.number_input(
        "Tavily Query Cap",
        min_value=1, max_value=1000,
        value=int(os.getenv("TAVILY_QUERY_CAP", "200")),
        key="tavily_cap_input",
    )

    st.markdown("<hr class='verdict-hr'>", unsafe_allow_html=True)

    # Run button
    is_running = "runner_thread" in st.session_state and st.session_state["runner_thread"].is_alive()
    run_disabled = is_running or uploaded_file is None or not gemini_key

    if st.button(
        "🚀 Run VERDICT" if not is_running else "⏳ Pipeline Running...",
        disabled=run_disabled,
        key="run_button",
        use_container_width=True,
        type="primary",
    ):
        _launch_pipeline(uploaded_file, gemini_key, tavily_key, global_model,
                        decomposer_model, synthesizer_model, embed_model, tavily_cap)

    if is_running:
        st.progress(0.5, text="Pipeline is running...")

    if "runner_thread" in st.session_state and not st.session_state["runner_thread"].is_alive():
        st.success("✅ Pipeline complete!")
        if st.button("Clear & Start New Run", key="clear_btn"):
            for key in ["runner_thread", "paper_id", "pdf_path", "status_messages"]:
                st.session_state.pop(key, None)
            st.rerun()

    # Status log
    if "status_messages" in st.session_state and st.session_state["status_messages"]:
        st.markdown("### 📋 Pipeline Log")
        with st.container(height=200):
            for msg in st.session_state["status_messages"][-30:]:
                st.markdown(f'<div class="log-entry">{msg}</div>', unsafe_allow_html=True)





# ─────────────────────────────────────────────────────────────────────────────
# Main display area
# ─────────────────────────────────────────────────────────────────────────────

def main():
    paper_id = st.session_state.get("paper_id")
    is_running = "runner_thread" in st.session_state and st.session_state["runner_thread"].is_alive()

    if paper_id is None:
        # Landing state
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#718096;">
            <div style="font-size:4rem;margin-bottom:1rem;">📤</div>
            <h3 style="color:#90cdf4;margin-bottom:0.5rem;">Upload a Research Paper to Begin</h3>
            <p>Upload a PDF in the sidebar, provide your API keys, and click <strong>Run VERDICT</strong></p>
            <p style="font-size:0.85rem;margin-top:1rem;">
                The pipeline will extract claims, decompose them into sub-hypotheses,<br>
                search for evidence, and produce belief-weighted verdicts.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Architecture diagram
        with st.expander("📐 How VERDICT Works", expanded=False):
            st.markdown("""
            ```
            PDF → Agent 1 (Claim Extractor)
                   ↓ per claim (ProcessPoolExecutor, 8-min timeout)
                  Agent 2 (Hypothesis Decomposer — never sees paper)
                   ↓ per sub-hypothesis
                  Agent 3 (Evidence Hunter: Tavily + PwC + ChromaDB RAG)
                  Agent 5 (Devil's Advocate: adversarial Tavily queries)
                   ↓ combined + deduplicated
                  Agent 4 (Evidence Judge: p-value + formal t-test)
                   ↓
                  SPRT (kappa calibrator, sequential multiplication)
                  Dempster-Shafer (belief triplet combination)
                  Disagreement Score (inter-agent variance)
                   ↓
                  Agent 6 (Verdict Synthesizer)
                   ↓
                  Dependency DAG (NetworkX topological propagation)
                   ↓
                  Paper-Level Summary
            ```
            """)

    else:
        # Active or completed run
        conn = _get_db_conn()
        try:
            claims = conn.execute(
                "SELECT * FROM claims WHERE paper_id=? ORDER BY position", (paper_id,)
            ).fetchall()
        except Exception:
            claims = []

        # Live Activity Log (shows granular node-level events across processes)
        try:
            ui_logs = conn.execute(
                "SELECT message FROM ui_logs WHERE paper_id=? ORDER BY id DESC LIMIT 15", (paper_id,)
            ).fetchall()
            if is_running and ui_logs:
                st.markdown('<div class="section-header">⚡ Live Pipeline Activity</div>', unsafe_allow_html=True)
                with st.container(height=180):
                    for log in reversed(ui_logs):
                        st.markdown(
                            f'<div style="font-family:monospace;font-size:0.85rem;color:#a0aec0;margin-bottom:4px;">→ {log["message"]}</div>', 
                            unsafe_allow_html=True
                        )
                st.markdown("<hr class='verdict-hr'>", unsafe_allow_html=True)
        except Exception:
            pass

        # Progress header
        if claims:
            done_count = sum(1 for c in claims if c["status"] in ("done", "failed", "timed_out"))
            total_count = len(claims)
            progress = done_count / total_count if total_count > 0 else 0
            st.progress(progress, text=f"Claims processed: {done_count}/{total_count}")

        # Claims panel
        if claims:
            st.markdown('<div class="section-header">📋 Claims</div>', unsafe_allow_html=True)
            for claim in claims:
                _render_claim_card(claim, conn)
        elif is_running:
            st.info("⏳ Extracting claims from paper...")

        # Paper summary (rendered when pipeline is done)
        if not is_running and paper_id:
            summary = conn.execute(
                "SELECT * FROM paper_summaries WHERE paper_id=? ORDER BY id DESC LIMIT 1",
                (paper_id,)
            ).fetchone()
            if summary:
                _render_paper_summary(summary, conn)

        conn.close()

        # Polling loop
        if is_running:
            time.sleep(2)
            st.rerun()

        elif "runner_thread" in st.session_state and not st.session_state["runner_thread"].is_alive():
            # Thread finished — clean up and do a final rerun to render summary
            del st.session_state["runner_thread"]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_claim_card(claim, conn: sqlite3.Connection) -> None:
    status = claim["status"]
    status_class = {
        "done": "badge-done", "running": "badge-running",
        "pending": "badge-pending", "failed": "badge-failed",
        "timed_out": "badge-timed_out",
    }.get(status, "badge-pending")

    claim_label = f"[{claim['type']}] {claim['text'][:80]}{'...' if len(claim['text']) > 80 else ''}"

    with st.expander(claim_label, expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f'<span class="badge {status_class}">{status}</span>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span class="metric-val">Weight: {claim["epistemic_weight"]:.2f}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<span class="metric-val">Section: {claim["section"] or "—"}</span>', unsafe_allow_html=True)

        st.markdown(f"**Claim:** {claim['text']}")

        # Verdict
        verdict = conn.execute(
            "SELECT * FROM verdicts WHERE claim_id=?", (claim["id"],)
        ).fetchone()

        if verdict:
            _render_belief_bar(
                float(verdict["support"] or 0),
                float(verdict["contradiction"] or 0),
                float(verdict["uncertainty"] or 1),
                label="Claim Verdict"
            )

            if verdict["conflict_flag"]:
                st.markdown(
                    '<span class="badge badge-conflict">⚠️ HIGH CONFLICT — VERDICT RELIABILITY LIMITED</span>',
                    unsafe_allow_html=True
                )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Support", f"{float(verdict['support'] or 0):.1%}")
            with col_b:
                st.metric("Contradiction", f"{float(verdict['contradiction'] or 0):.1%}")
            with col_c:
                adj = float(verdict['adjusted_support'] or 0)
                st.metric("Adj. Support", f"{adj:.1%}",
                         help="Support score after dependency propagation")

            if verdict["disagreement_score"] is not None:
                st.markdown(
                    f'<span class="metric-val">Inter-agent disagreement: {float(verdict["disagreement_score"]):.4f}</span>',
                    unsafe_allow_html=True
                )

            if verdict["plain_language"]:
                st.markdown("**📝 Verdict:**")
                st.info(verdict["plain_language"])

        # Sub-hypotheses
        sub_hyps = conn.execute(
            "SELECT * FROM sub_hypotheses WHERE claim_id=? ORDER BY position",
            (claim["id"],)
        ).fetchall()

        if sub_hyps:
            st.markdown("**🔬 Sub-Hypotheses:**")
            for sh in sub_hyps:
                with st.expander(f"Sub-H {sh['position']}: {sh['text'][:60]}...", expanded=False):
                    st.markdown(f"*Logical relationship:* {sh['logical_relationship'] or '—'}")
                    st.markdown(f"**{sh['text']}**")

                    # DS masses
                    ds = conn.execute(
                        "SELECT * FROM ds_masses WHERE sub_hyp_id=?", (sh["id"],)
                    ).fetchone()
                    if ds:
                        _render_belief_bar(
                            float(ds["support"] or 0),
                            float(ds["contradiction"] or 0),
                            float(ds["uncertainty"] or 1),
                            label="DS Belief"
                        )
                        if ds["conflict_flag"]:
                            st.markdown(
                                '<span class="badge badge-conflict">⚠️ HIGH CONFLICT</span>',
                                unsafe_allow_html=True
                            )

                    # SPRT
                    sprt = conn.execute(
                        "SELECT * FROM sprt_results WHERE sub_hyp_id=?", (sh["id"],)
                    ).fetchone()
                    if sprt:
                        decision = sprt["decision"] or "UNDECIDED"
                        dec_class = {
                            "FALSIFIED": "badge-FALSIFIED",
                            "INSUFFICIENT_EVIDENCE": "badge-INSUFFICIENT",
                            "UNDECIDED": "badge-UNDECIDED",
                        }.get(decision, "badge-UNDECIDED")
                        st.markdown(
                            f'SPRT Decision: <span class="badge {dec_class}">{decision}</span> '
                            f'<span class="metric-val">(product={float(sprt["product"] or 1):.4f})</span>',
                            unsafe_allow_html=True
                        )

                        # Step log table
                        try:
                            step_log = json.loads(sprt["step_log"] or "[]")
                            if step_log:
                                import pandas as pd
                                df = pd.DataFrame(step_log)
                                df.columns = [c.upper() for c in df.columns]
                                st.dataframe(df, use_container_width=True, height=min(200, len(df)*38+40))
                        except Exception:
                            pass

                    # Evidence list
                    evidence = conn.execute(
                        "SELECT * FROM evidence WHERE sub_hyp_id=?", (sh["id"],)
                    ).fetchall()
                    if evidence:
                        st.markdown(f"**🔍 Evidence ({len(evidence)} items):**")
                        for ev in evidence:
                            tier_emoji = {"peer_reviewed": "📰", "preprint": "📄", "blog": "🌐"}.get(
                                ev["reliability_tier"], "🔗"
                            )
                            tag = ev["p_value_tag"] or "approximate"
                            tag_class = "badge-formal" if tag == "formal" else "badge-approx"
                            direc = ev["directionality"] or "inconclusive"
                            direc_emoji = {"supporting": "✅", "contradicting": "❌", "inconclusive": "❓"}.get(direc, "❓")

                            st.markdown(
                                f"{tier_emoji} [{ev['reliability_tier']}] "
                                f"{direc_emoji} **{direc}** | "
                                f"p={float(ev['p_value'] or 0.5):.3f} "
                                f'<span class="badge {tag_class}">{tag}</span> | '
                                f"Agent: {ev['agent_source'] or '—'}",
                                unsafe_allow_html=True
                            )
                            with st.expander(f"Source: {str(ev['source'] or '')[:60]}", expanded=False):
                                st.markdown(ev["content"] or "—")


def _render_belief_bar(support: float, contradiction: float, uncertainty: float, label: str = "") -> None:
    """Render a horizontal stacked belief bar."""
    s_pct = int(support * 100)
    c_pct = int(contradiction * 100)
    u_pct = 100 - s_pct - c_pct

    if label:
        st.markdown(f'<div style="color:#a0aec0;font-size:0.8rem;margin-bottom:0.25rem;">{label}</div>',
                   unsafe_allow_html=True)

    st.markdown(f"""
    <div class="belief-bar-container">
        <div class="belief-support" style="width:{s_pct}%" title="Support {s_pct}%"></div>
        <div class="belief-contra" style="width:{c_pct}%" title="Contradiction {c_pct}%"></div>
        <div class="belief-uncertain" style="width:{u_pct}%" title="Uncertainty {u_pct}%"></div>
    </div>
    <div style="display:flex;gap:1rem;font-size:0.72rem;color:#718096;margin-bottom:0.5rem;">
        <span style="color:#68d391;">■ Support {s_pct}%</span>
        <span style="color:#fc8181;">■ Contradiction {c_pct}%</span>
        <span style="color:#a0aec0;">■ Uncertainty {u_pct}%</span>
    </div>
    """, unsafe_allow_html=True)


def _render_paper_summary(summary, conn: sqlite3.Connection) -> None:
    """Render the paper-level summary panel."""
    st.markdown('<div class="section-header">📊 Paper-Level Summary</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)

        # Overall belief
        st.markdown("#### Overall Belief Distribution")
        _render_belief_bar(
            float(summary["overall_support"] or 0),
            float(summary["overall_contradiction"] or 0),
            float(summary["overall_uncertainty"] or 1),
            label="Across all claims"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💪 Top 3 Strongest Claims")
            try:
                strong = json.loads(summary["top_strong_claims"] or "[]")
                for item in strong:
                    cid = item["claim_id"]
                    claim_row = conn.execute("SELECT text FROM claims WHERE id=?", (cid,)).fetchone()
                    if claim_row:
                        st.markdown(
                            f"- **{item['adjusted_support']:.1%}** support — "
                            f"{claim_row['text'][:100]}..."
                        )
            except Exception:
                pass

        with col2:
            st.markdown("#### ⚠️ Top 3 Weakest Claims")
            try:
                weak = json.loads(summary["top_weak_claims"] or "[]")
                for item in weak:
                    cid = item["claim_id"]
                    claim_row = conn.execute("SELECT text FROM claims WHERE id=?", (cid,)).fetchone()
                    if claim_row:
                        st.markdown(
                            f"- Weakness score **{item['weakness_score']:.2f}** — "
                            f"{claim_row['text'][:100]}..."
                        )
            except Exception:
                pass

        # Dependency summary
        if summary["dependency_summary"]:
            st.markdown("#### 🔗 Dependency Graph")
            st.markdown(summary["dependency_summary"])

        # Formally supported vs implied
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### ✅ Formally Supported")
            try:
                formal = json.loads(summary["formally_supported"] or "[]")
                if formal:
                    for f in formal:
                        st.markdown(f"- {f['text'][:100]}...")
                else:
                    st.markdown("*No claims meet the formal support threshold (>60% adjusted support).*")
            except Exception:
                pass

        with col4:
            st.markdown("#### 💭 Implied but Not Proven")
            try:
                implied = json.loads(summary["implied_only"] or "[]")
                if implied:
                    for im in implied[:5]:
                        st.markdown(f"- {im['text'][:100]}...")
                    if len(implied) > 5:
                        st.markdown(f"*...and {len(implied)-5} more.*")
                else:
                    st.markdown("*All claims meet formal support threshold.*")
            except Exception:
                pass

        # Reproducibility flag
        if summary["reproducibility_flag"]:
            st.markdown("""
            <div class="repro-flag">
                ⚠️ <strong>Reproducibility Concern Detected</strong><br>
                One or more claims reference proprietary data, internal datasets,
                or data available only upon request. External validation may be limited.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '🟢 **Reproducibility:** No obvious reproducibility concerns detected in claim language.',
            )

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
