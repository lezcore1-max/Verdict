"""
orchestration/graph.py — LangGraph state graph for per-claim pipeline.

Each node corresponds to one agent or the math computation step.
Agent 2 (decompose)
    ↓
Agent 3 ──┐
          ├── merge + dedup → Agent 4 (judge)
Agent 5 ──┘
    ↓
Math (SPRT + DST)
    ↓
Agent 6 (synthesize)
Conditional edges skip to A6 on extraction failure.
"""
import logging
from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────────────────────────────────────

class VerdictState(TypedDict, total=False):
    # Inputs
    paper_id: int
    claim_id: int
    claim_text: str
    claim_type: str
    model_name: str
    api_key: Optional[str]
    tavily_key: Optional[str]
    db_path: str
    chroma_dir: str

    # Agent 2 output
    sub_hypotheses: list[dict]   # list of SubHypothesis dicts

    # Agent 3 & 5 output (per sub-hypothesis)
    evidence_per_sub: dict[int, list[dict]]    # position → list[EvidenceItem dicts]
    judged_per_sub: dict[int, list[dict]]      # position → list[JudgedEvidence dicts]

    # Math results (per sub-hypothesis)
    sprt_per_sub: dict[int, dict]              # position → SPRT result dict
    ds_per_sub: dict[int, dict]                # position → DS mass dict
    conflict_flags: dict[int, bool]
    disagreement_scores: dict[int, float]

    # Agent 6 output
    final_verdict: Optional[dict]

    # Error tracking
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Node functions
# ─────────────────────────────────────────────────────────────────────────────

def _log_ui(state: VerdictState, msg: str):
    import sqlite3
    from core.database import insert_ui_log
    try:
        conn = sqlite3.connect(state["db_path"], check_same_thread=False)
        c_text = state.get("claim_text", "")
        c_str = f"{c_text[:30]}..." if len(c_text) > 30 else c_text
        insert_ui_log(conn, state["paper_id"], f"[Claim {state['claim_id']}] {c_str} - {msg}")
        conn.close()
    except Exception:
        pass

def node_decompose(state: VerdictState) -> VerdictState:
    """Agent 2: decompose claim into sub-hypotheses."""
    _log_ui(state, "Agent 2: Decomposing into sub-hypotheses...")
    import agents.agent2_hypothesis_decomposer as a2
    result = a2.run(
        claim_text=state["claim_text"],
        claim_type=state["claim_type"],
        model_name=state.get("model_name", "gemini-2.0-flash"),
        api_key=state.get("api_key"),
    )
    if result is None:
        return {**state, "error": "EXTRACTION_FAILED:agent2", "sub_hypotheses": []}
    subs = [sh.model_dump() for sh in result.sub_hypotheses]
    return {**state, "sub_hypotheses": subs, "error": None}


def node_hunt_evidence(state: VerdictState) -> VerdictState:
    """Agent 3 + Agent 5: retrieve and deduplicate evidence per sub-hypothesis."""
    _log_ui(state, "Agents 3 & 5: Hunting evidence across web & RAG...")
    import sqlite3
    from core.config import CHROMA_DIR
    from core.rag import get_chroma_collection
    import agents.agent3_evidence_hunter as a3
    import agents.agent5_devils_advocate as a5
    from core.embedder import LocalEmbedder
    from core.config import COSINE_DEDUP

    db_conn = sqlite3.connect(state["db_path"], check_same_thread=False)
    db_conn.execute("PRAGMA journal_mode=WAL")

    chroma_dir = state.get("chroma_dir", CHROMA_DIR)
    try:
        collection = get_chroma_collection(chroma_dir)
    except Exception:
        collection = None

    evidence_per_sub: dict[int, list[dict]] = {}

    for sh in state.get("sub_hypotheses", []):
        pos = sh["position"]
        sub_text = sh["text"]

        # Agent 3
        ev3 = a3.run(
            sub_hyp_text=sub_text,
            claim_type=state["claim_type"],
            chroma_collection=collection,
            paper_id=state["paper_id"],
            db_conn=db_conn,
            model_name=state.get("model_name", "gemini-2.0-flash"),
            api_key=state.get("api_key"),
            tavily_key=state.get("tavily_key"),
        )

        # Agent 5
        ev5 = a5.run(
            sub_hyp_text=sub_text,
            paper_id=state["paper_id"],
            db_conn=db_conn,
            model_name=state.get("model_name", "gemini-2.0-flash"),
            api_key=state.get("api_key"),
            tavily_key=state.get("tavily_key"),
        )

        # Merge and deduplicate across both agents BEFORE any math
        all_ev = ev3.evidence + ev5.evidence
        embedder = LocalEmbedder()
        from agents.agent3_evidence_hunter import _doi_arxiv_key
        deduped = embedder.deduplicate(
            all_ev,
            key_fn=lambda e: e.content,
            threshold=COSINE_DEDUP,
            exact_key_fn=_doi_arxiv_key,
        )
        evidence_per_sub[pos] = [e.model_dump() for e in deduped]

    db_conn.close()
    return {**state, "evidence_per_sub": evidence_per_sub}


def node_judge_evidence(state: VerdictState) -> VerdictState:
    """Agent 4: judge each evidence item per sub-hypothesis."""
    _log_ui(state, "Agent 4: Judging evidence strength and computing p-values...")
    import agents.agent4_evidence_judge as a4
    from agents.schemas import EvidenceItem

    judged_per_sub: dict[int, list[dict]] = {}

    for sh in state.get("sub_hypotheses", []):
        pos = sh["position"]
        sub_text = sh["text"]
        ev_list = state.get("evidence_per_sub", {}).get(pos, [])

        judged_list = []
        for ev_dict in ev_list:
            ev_item = EvidenceItem.model_validate(ev_dict)
            judged = a4.run(
                sub_hyp_text=sub_text,
                evidence=ev_item,
                claim_type=state["claim_type"],
                model_name=state.get("model_name", "gemini-2.0-flash"),
                api_key=state.get("api_key"),
            )
            if judged is not None:
                d = judged.model_dump()
                d["agent_source"] = ev_item.agent_source
                judged_list.append(d)

        judged_per_sub[pos] = judged_list

    return {**state, "judged_per_sub": judged_per_sub}


def node_run_math(state: VerdictState) -> VerdictState:
    """SPRT + Dempster-Shafer math on combined deduplicated evidence."""
    _log_ui(state, "Math Engine: Running SPRT and Dempster-Shafer combinations...")
    import numpy as np
    from verdict_math.sprt import run_sprt
    from verdict_math.dempster_shafer import (
        evidence_to_mass, combine_all, add_disagreement, compute_disagreement_score
    )
    from core.config import P_VALUE_FLOOR

    sprt_per_sub: dict[int, dict] = {}
    ds_per_sub: dict[int, dict] = {}
    conflict_flags: dict[int, bool] = {}
    disagreement_scores: dict[int, float] = {}

    for sh in state.get("sub_hypotheses", []):
        pos = sh["position"]
        judged_list = state.get("judged_per_sub", {}).get(pos, [])

        if not judged_list:
            sprt_per_sub[pos] = {"step_log": [], "product": 1.0, "decision": "UNDECIDED"}
            ds_per_sub[pos] = {"support": 0.0, "contradiction": 0.0, "uncertainty": 1.0}
            conflict_flags[pos] = False
            disagreement_scores[pos] = 0.0
            continue

        # Extract p-values and tags for SPRT
        p_values = [max(float(j["p_value"]), P_VALUE_FLOOR) for j in judged_list]
        tags = [j["p_value_tag"] for j in judged_list]

        # Run SPRT
        sprt_result = run_sprt(p_values, tags)
        sprt_per_sub[pos] = sprt_result

        # Dempster-Shafer masses from each evidence item
        masses = [
            evidence_to_mass(
                p_value=max(float(j["p_value"]), P_VALUE_FLOOR),
                directionality=j["directionality"],
                directness=j.get("directness", "partial_test"),
            )
            for j in judged_list
        ]

        combined, conflict = combine_all(masses)
        conflict_flags[pos] = conflict

        # Inter-agent disagreement: variance between agent4 and agent5 p-values
        a4_p = [float(j["p_value"]) for j in judged_list if j.get("agent_source") == "agent3"]
        a5_p = [float(j["p_value"]) for j in judged_list if j.get("agent_source") == "agent5"]
        disagreement = compute_disagreement_score(a4_p, a5_p)
        disagreement_scores[pos] = disagreement

        # Add disagreement to uncertainty
        final_triplet = add_disagreement(combined, disagreement)

        ds_per_sub[pos] = {
            "support": float(final_triplet[0]),
            "contradiction": float(final_triplet[1]),
            "uncertainty": float(final_triplet[2]),
        }

    return {
        **state,
        "sprt_per_sub": sprt_per_sub,
        "ds_per_sub": ds_per_sub,
        "conflict_flags": conflict_flags,
        "disagreement_scores": disagreement_scores,
    }


def node_synthesize(state: VerdictState) -> VerdictState:
    """Agent 6: synthesise final verdict."""
    _log_ui(state, "Agent 6: Synthesizing final claim verdict...")
    import agents.agent6_verdict_synthesizer as a6

    sub_hyps = state.get("sub_hypotheses", [])
    sprt = state.get("sprt_per_sub", {})
    ds = state.get("ds_per_sub", {})
    cflags = state.get("conflict_flags", {})
    disc = state.get("disagreement_scores", {})

    sub_verdicts = [{"text": sh["text"]} for sh in sub_hyps]
    sprt_list = [sprt.get(sh["position"], {}) for sh in sub_hyps]
    ds_list = [ds.get(sh["position"], {}) for sh in sub_hyps]
    flags_list = [cflags.get(sh["position"], False) for sh in sub_hyps]
    disagreement_score = float(
        sum(disc.get(sh["position"], 0.0) for sh in sub_hyps) / max(len(sub_hyps), 1)
    )

    verdict = a6.run(
        claim_text=state.get("claim_text", ""),
        sub_hyp_verdicts=sub_verdicts,
        sprt_results=sprt_list,
        ds_masses=ds_list,
        conflict_flags=flags_list,
        disagreement_score=disagreement_score,
        model_name=state.get("model_name", "gemini-2.0-flash"),
        api_key=state.get("api_key"),
    )

    if verdict is None:
        return {**state, "final_verdict": None, "error": "EXTRACTION_FAILED:agent6"}

    return {**state, "final_verdict": verdict.model_dump(), "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edges
# ─────────────────────────────────────────────────────────────────────────────

def should_skip_to_synthesize(state: VerdictState) -> str:
    if state.get("error") or not state.get("sub_hypotheses"):
        return "synthesize"
    return "hunt_evidence"


def after_hunt(state: VerdictState) -> str:
    if state.get("error"):
        return "synthesize"
    return "judge_evidence"


def after_judge(state: VerdictState) -> str:
    return "run_math"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(VerdictState)

    g.add_node("decompose", node_decompose)
    g.add_node("hunt_evidence", node_hunt_evidence)
    g.add_node("judge_evidence", node_judge_evidence)
    g.add_node("run_math", node_run_math)
    g.add_node("synthesize", node_synthesize)

    g.set_entry_point("decompose")
    g.add_conditional_edges("decompose", should_skip_to_synthesize)
    g.add_conditional_edges("hunt_evidence", after_hunt)
    g.add_edge("judge_evidence", "run_math")
    g.add_edge("run_math", "synthesize")
    g.add_edge("synthesize", END)

    return g


def compile_graph():
    return build_graph().compile()
