"""
smoke_test.py — Verify all core math and schema imports work correctly.
Run with: python smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("VERDICT Smoke Tests")
print("=" * 60)

errors = []

# ── Test 1: SPRT math ─────────────────────────────────────────────────────────
print("\n[1] SPRT math...")
try:
    from verdict_math.sprt import kappa_e_value, run_sprt
    import math as _math

    # kappa_e_value(0.04) = 0.5 * 0.04^(-0.5) = 0.5 * 5.0 = 2.5
    e = kappa_e_value(0.04)
    assert abs(e - 2.5) < 1e-9, f"Expected 2.5, got {e}"

    # p=0.04 three times → product = 2.5^3 = 15.625 → FALSIFIED (>=8.0)
    result = run_sprt([0.04, 0.04, 0.04])
    assert result["decision"] == "FALSIFIED", f"Expected FALSIFIED, got {result['decision']}"
    assert abs(result["product"] - 15.625) < 0.001, f"Expected 15.625, got {result['product']}"

    # All p=0.5 → e=0.5*0.5^(-0.5)=0.5*1.414=0.707 → product keeps decreasing → INSUFFICIENT
    result2 = run_sprt([0.5] * 20)
    assert result2["decision"] == "INSUFFICIENT_EVIDENCE", f"Got {result2['decision']}"

    print("  ✅ SPRT: kappa_e_value, FALSIFIED, INSUFFICIENT_EVIDENCE all correct")
except Exception as exc:
    print(f"  ❌ SPRT failed: {exc}")
    errors.append(("SPRT", str(exc)))


# ── Test 2: Dempster-Shafer ───────────────────────────────────────────────────
print("\n[2] Dempster-Shafer...")
try:
    import numpy as np
    from verdict_math.dempster_shafer import (
        evidence_to_mass, combine_two, combine_all, add_disagreement,
        compute_disagreement_score, ConflictError
    )

    # evidence_to_mass: supporting, p=0.1 → [0.9, 0.0, 0.1]
    m = evidence_to_mass(0.1, "supporting", "direct_test")
    assert abs(m[0] - 0.9) < 1e-9 and abs(m[1]) < 1e-9 and abs(m[2] - 0.1) < 1e-9, f"Got {m}"

    # tangential → [0,0,1]
    m2 = evidence_to_mass(0.01, "supporting", "tangential")
    assert np.allclose(m2, [0, 0, 1]), f"Got {m2}"

    # inconclusive → [0,0,1]
    m3 = evidence_to_mass(0.05, "inconclusive", "direct_test")
    assert np.allclose(m3, [0, 0, 1]), f"Got {m3}"

    # Case 1: A=(0.5,0.1,0.4), B=(0.1,0.5,0.4) → K=0.74
    ma = np.array([0.5, 0.1, 0.4])
    mb = np.array([0.1, 0.5, 0.4])
    result = combine_two(ma, mb)
    assert abs(result[0] - 0.3919) < 0.001, f"Support mismatch: {result}"
    assert abs(result[1] - 0.3919) < 0.001, f"Contradiction mismatch: {result}"
    assert abs(result[2] - 0.2162) < 0.001, f"Uncertainty mismatch: {result}"
    assert abs(result.sum() - 1.0) < 1e-9, f"Does not sum to 1: {result}"

    # Case 2: ConflictError when K < 0.1
    mc = np.array([0.95, 0.05, 0.0])
    md = np.array([0.05, 0.95, 0.0])
    try:
        combine_two(mc, md)
        raise AssertionError("Should have raised ConflictError")
    except ConflictError as e:
        pass  # Expected

    # combine_all: conflict halts at last_good
    _, flag = combine_all([mc, md])
    assert flag, "Expected conflict_flag=True"

    _, flag_ok = combine_all([ma, mb])
    assert not flag_ok, "Expected no conflict"

    # add_disagreement
    triplet = np.array([0.6, 0.2, 0.2])
    adjusted = add_disagreement(triplet, 0.1)
    assert abs(adjusted.sum() - 1.0) < 1e-9, f"Does not sum to 1: {adjusted}"
    assert adjusted[2] > triplet[2], "Uncertainty should increase"

    # compute_disagreement_score
    score = compute_disagreement_score([0.1, 0.2], [0.8, 0.9])
    assert score > 0, "Variance should be positive for different agents"

    print("  ✅ Dempster-Shafer: all 8 sub-tests passed")
except Exception as exc:
    print(f"  ❌ Dempster-Shafer failed: {exc}")
    errors.append(("DS", str(exc)))


# ── Test 3: Dependency graph ──────────────────────────────────────────────────
print("\n[3] Dependency graph...")
try:
    from verdict_math.dependency_graph import build_dag, propagate_scores

    dag = build_dag([(1, 2), (2, 3)])
    scores = {1: 0.4, 2: 0.8, 3: 0.9}
    adj = propagate_scores(dag, scores)

    # Node 1 is root → unchanged
    assert abs(adj[1] - 0.4) < 1e-9, f"Root should be unchanged: {adj[1]}"
    # Node 2 depends on 1 → 0.8 * 0.4 = 0.32
    assert abs(adj[2] - 0.32) < 1e-9, f"Expected 0.32, got {adj[2]}"
    # Node 3 depends on 2 → 0.9 * 0.32 = 0.288
    assert abs(adj[3] - 0.288) < 1e-9, f"Expected 0.288, got {adj[3]}"

    # Cycle detection: adding 3→1 should be dropped
    dag2 = build_dag([(1, 2), (2, 3), (3, 1)])
    import networkx as nx
    assert nx.is_directed_acyclic_graph(dag2), "Cycle should have been removed"

    print("  ✅ Dependency graph: propagation and cycle detection correct")
except Exception as exc:
    print(f"  ❌ Dependency graph failed: {exc}")
    errors.append(("DepGraph", str(exc)))


# ── Test 4: Pydantic schemas ──────────────────────────────────────────────────
print("\n[4] Pydantic schemas...")
try:
    from agents.schemas import (
        ExtractedClaim, ClaimExtractOutput, SubHypothesis, HypothesisDecompOutput,
        EvidenceItem, JudgedEvidence, ClaimVerdict
    )

    # JudgedEvidence clamps p_value
    j = JudgedEvidence(
        directly_tests=True,
        directionality="supporting",
        strength="strong",
        p_value=0.0,  # should be clamped to 1e-6
        p_value_tag="approximate",
    )
    assert j.p_value >= 1e-6, f"p_value not clamped: {j.p_value}"

    # ClaimVerdict normalises
    v = ClaimVerdict(support=3.0, contradiction=1.0, uncertainty=0.0,
                    plain_language="test")
    assert abs(v.support + v.contradiction + v.uncertainty - 1.0) < 1e-9

    print("  ✅ Pydantic schemas: validation and clamping correct")
except Exception as exc:
    print(f"  ❌ Schemas failed: {exc}")
    errors.append(("Schemas", str(exc)))


# ── Test 5: SQLite database ───────────────────────────────────────────────────
print("\n[5] SQLite database...")
try:
    import sqlite3
    from core.database import init_db, get_connection, upsert_paper, atomic_increment_tavily_counter

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    # All 11 tables should exist
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    expected = {
        "papers", "claims", "dependency_edges", "sub_hypotheses", "evidence",
        "sprt_results", "ds_masses", "verdicts", "agent_logs", "pipeline_state",
        "paper_summaries"
    }
    missing = expected - tables
    assert not missing, f"Missing tables: {missing}"

    # Atomic Tavily counter
    conn.execute("INSERT INTO papers(id,file_path,status,tavily_count) VALUES(1,'x','pending',0)")
    conn.commit()
    assert atomic_increment_tavily_counter(conn, 1, cap=2) == True   # count→1
    assert atomic_increment_tavily_counter(conn, 1, cap=2) == True   # count→2
    assert atomic_increment_tavily_counter(conn, 1, cap=2) == False  # cap reached

    print("  ✅ SQLite: all 11 tables present, atomic counter correct")
except Exception as exc:
    print(f"  ❌ SQLite failed: {exc}")
    errors.append(("SQLite", str(exc)))


# ── Test 6: PDF parser ────────────────────────────────────────────────────────
print("\n[6] PDF parser...")
try:
    from core.pdf_parser import chunk_text, detect_section, epistemic_weight_for_section

    chunks = chunk_text("word " * 1000, chunk_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1, "Should produce multiple chunks"
    assert all(len(c.split()) <= 110 for c in chunks), "Chunks too large"

    sec = detect_section("Abstract\nThis paper presents...")
    assert sec == "abstract", f"Expected abstract, got {sec}"

    w = epistemic_weight_for_section("results")
    assert w > 0.7, f"Results should have high weight, got {w}"

    print("  ✅ PDF parser: chunking and section detection correct")
except Exception as exc:
    print(f"  ❌ PDF parser failed: {exc}")
    errors.append(("PDFParser", str(exc)))


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"❌ {len(errors)} test(s) FAILED:")
    for name, msg in errors:
        print(f"   {name}: {msg}")
    sys.exit(1)
else:
    print("✅ All 6 smoke tests PASSED")
    print("=" * 60)
    print("\nNext step: streamlit run frontend/app.py")
