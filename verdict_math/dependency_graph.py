"""
math/dependency_graph.py — NetworkX DAG construction and score propagation.

Dependency semantics:
    Edge A → B means "B depends on A" (A is a methodological predecessor of B).
    Propagation: B's adjusted support = B's raw support × min(support of all predecessors).
    This propagates foundational weaknesses upward through the claim graph.
"""
import logging
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def build_dag(edges: list[tuple[int, int]]) -> nx.DiGraph:
    """
    Build a directed acyclic graph from (from_claim_id, to_claim_id) pairs.

    Cycles are detected and the offending edge is dropped with a warning
    to ensure topological sort is always possible.
    """
    dag = nx.DiGraph()
    for (from_id, to_id) in edges:
        dag.add_edge(from_id, to_id)
        # Check for cycle after each addition; remove edge if cycle created
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(from_id, to_id)
            logger.warning(
                "Removed dependency edge (%d → %d) because it creates a cycle",
                from_id, to_id
            )
    return dag


def propagate_scores(
    dag: nx.DiGraph,
    support_scores: dict[int, float],
) -> dict[int, float]:
    """
    Propagate foundational weaknesses through the dependency graph.

    Algorithm (from scratch — no library function for this):
        For each node in topological order (roots first):
            predecessors = dag.predecessors(node)
            if predecessors exist:
                min_pred_support = min(support_scores[p] for p in predecessors)
                support_scores[node] *= min_pred_support
            # else: root node — score unchanged

    Returns a new dict with adjusted support scores.

    Why topological order?  It guarantees that when we process node B, all
    predecessor nodes A have already been processed (possibly adjusted by
    their own predecessors), so the propagation is transitive.
    """
    adjusted = dict(support_scores)  # copy; do not mutate input

    for node in nx.topological_sort(dag):
        preds = list(dag.predecessors(node))
        if not preds:
            continue  # root — no adjustment

        # All predecessors must be in our score dict; fall back to 1.0 if missing
        pred_supports = [adjusted.get(p, 1.0) for p in preds]
        min_pred_support = float(np.min(pred_supports))

        adjusted[node] = float(adjusted.get(node, 1.0)) * min_pred_support

    return adjusted


def build_dependency_summary(
    dag: nx.DiGraph,
    adjusted_scores: dict[int, float],
    threshold: float = 0.3,
) -> str:
    """
    Generate a plain-language summary of the dependency graph.

    Reports:
      - Total number of dependency edges
      - Number of claims with predecessors (non-roots)
      - Whether any predecessor has adjusted_support < threshold
    """
    n_edges = dag.number_of_edges()
    n_dependent = sum(1 for n in dag.nodes() if list(dag.predecessors(n)))
    weak_predecessors = [
        n for n in dag.nodes()
        if adjusted_scores.get(n, 1.0) < threshold
    ]

    lines = [
        f"The dependency graph contains {n_edges} dependency edge(s) "
        f"across {dag.number_of_nodes()} claim(s).",
        f"{n_dependent} claim(s) depend on one or more predecessor claims.",
    ]

    if weak_predecessors:
        lines.append(
            f"⚠️ {len(weak_predecessors)} predecessor claim(s) have "
            f"adjusted support below {threshold:.0%} — "
            "this foundational weakness propagates to all dependent claims."
        )
    else:
        lines.append(
            "No foundational weaknesses detected in the dependency chain."
        )

    return " ".join(lines)
