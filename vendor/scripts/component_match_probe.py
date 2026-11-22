"""Validation-only probe of connected-component boundary matching.

No main-experiment files are changed.  Only the first sorted validation image
and its first two annotations are read.  The real prediction--GT radius graph
has disjoint connected components.  Maximum cardinality and minimum total
distance are additive over those components, so they can be optimized
independently.  The original global dummy penalty is retained in every block.
Tie choices may change the distribution/p95 of matched distances even when
cardinality and total distance are identical; the probe reports that explicitly.
"""

from pathlib import Path
import json
import time

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

import bsds_support as original


def component_match(pred, truth, tolerance=2.0, return_stats=False):
    """Original max-cardinality/min-distance objective, split into real components."""
    a = np.column_stack(np.nonzero(pred))
    b = np.column_stack(np.nonzero(truth))
    counts = (len(a), len(b))
    if not len(a) or not len(b):
        result = (np.empty(0), *counts)
        return (result, {"real_components": 0}) if return_stats else result
    pairs = cKDTree(a).sparse_distance_matrix(cKDTree(b), tolerance,
                                             output_type="coo_matrix")
    if not pairs.nnz:
        result = (np.empty(0), *counts)
        return (result, {"real_components": 0, "isolated_predictions": len(a)}) if return_stats else result
    # Explicit unit weights retain zero-distance real edges in the adjacency.
    adjacency = coo_matrix((np.ones(2 * pairs.nnz, dtype=np.uint8),
                           (np.r_[pairs.row, len(a) + pairs.col],
                            np.r_[len(a) + pairs.col, pairs.row])),
                          shape=(len(a) + len(b), len(a) + len(b))).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    edge_component = labels[pairs.row]
    order = np.argsort(edge_component, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(edge_component[order])) + 1, len(order)]
    penalty = (min(len(a), len(b)) + 1) * (tolerance + 1)
    prediction_indices, truth_indices = [], []
    sizes = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        edges = order[left:right]
        local_a, row = np.unique(pairs.row[edges], return_inverse=True)
        local_b, col = np.unique(pairs.col[edges], return_inverse=True)
        cost = pairs.data[edges] + 1e-8
        graph = coo_matrix((np.r_[cost, np.full(len(local_a), penalty)],
                            (np.r_[row, np.arange(len(local_a))],
                             np.r_[col, len(local_b) + np.arange(len(local_a))])),
                           shape=(len(local_a), len(local_b) + len(local_a))).tocsr()
        ia, ib = min_weight_full_bipartite_matching(graph)
        real = ib < len(local_b)
        prediction_indices.extend(local_a[ia[real]])
        truth_indices.extend(local_b[ib[real]])
        sizes.append((len(local_a), len(local_b), len(edges)))
    distances = np.linalg.norm(a[prediction_indices] - b[truth_indices], axis=1)
    result = (distances, *counts)
    stats = {
        "real_components": len(sizes), "real_edges": int(pairs.nnz),
        "isolated_predictions": int(len(a) - len(np.unique(pairs.row))),
        "largest_component_pred": max(x[0] for x in sizes),
        "largest_component_gt": max(x[1] for x in sizes),
        "largest_component_edges": max(x[2] for x in sizes),
    }
    return (result, stats) if return_stats else result


def _metrics(result):
    distances, n_pred, n_gt = result
    count = len(distances)
    precision = count / n_pred if n_pred else 0.0
    recall = count / n_gt if n_gt else 0.0
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                matched_mean_px=float(distances.mean()) if count else None,
                matched_p95_px=float(np.percentile(distances, 95)) if count else None,
                matched=count, predicted=n_pred, truth=n_gt,
                total_distance=float(distances.sum()))


def main():
    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parents[1]
    data = root.parent / "BSR_bsds500" / "BSR" / "BSDS500" / "data"
    path = sorted((data / "images" / "val").glob("*.jpg"))[0]
    _, _, image, _ = original.read_input(path)
    annotations = original.truths(data / "groundTruth" / "val" / f"{path.stem}.mat")[:2]
    detector = original.Detector(image)
    valid = np.zeros(image.shape, dtype=bool)
    valid[9:-9, 9:-9] = True
    rows = []
    started = time.perf_counter()
    for family in ("A0_detect", "A1_detect"):
        for high in (3.0, 5.0, 10.0):
            pred = detector.edge(family, high)
            for annotator, truth in enumerate(annotations, 1):
                before = time.perf_counter()
                baseline = original.fast_match(pred & valid, truth & valid)
                baseline_s = time.perf_counter() - before
                before = time.perf_counter()
                candidate, stats = component_match(pred & valid, truth & valid,
                                                   return_stats=True)
                component_s = time.perf_counter() - before
                old, new = _metrics(baseline), _metrics(candidate)
                same_count = (old["matched"] == new["matched"]
                              and baseline[1:] == candidate[1:])
                same_cost = bool(np.isclose(old["total_distance"], new["total_distance"],
                                            rtol=0, atol=1e-8))
                same_distances = (len(baseline[0]) == len(candidate[0])
                                  and np.array_equal(np.sort(baseline[0]), np.sort(candidate[0])))
                row = dict(image_id=path.stem, split="val", family=family, high=high,
                           annotator=annotator, baseline_seconds=baseline_s,
                           component_seconds=component_s,
                           speedup=baseline_s / component_s if component_s else None,
                           same_count=same_count, same_total_cost=same_cost,
                           same_sorted_distances=same_distances,
                           baseline=old, component=new, graph=stats)
                if not rows:
                    # Independently verify the public baseline wrapper uses
                    # the same common border and metric definitions.
                    public = original.evaluate(pred, truth)
                    row["public_evaluate_equal"] = all(public[k] == old[k] for k in public)
                    assert row["public_evaluate_equal"]
                rows.append(row)
                print(json.dumps(row), flush=True)
                assert same_count and same_cost, "Objective changed; do not use component method"
    output = root / "audit" / "component_match_probe.json"
    output.parent.mkdir(exist_ok=True)
    baseline_total = sum(r["baseline_seconds"] for r in rows)
    component_total = sum(r["component_seconds"] for r in rows)
    report = dict(validation_image=path.stem, reads_test=False,
                  original_source_sha256=original.digest(Path(original.__file__)),
                  probe_source_sha256=original.digest(Path(__file__)),
                  comparisons=len(rows), all_counts_equal=all(r["same_count"] for r in rows),
                  all_total_costs_equal=all(r["same_total_cost"] for r in rows),
                  all_sorted_distances_equal=all(r["same_sorted_distances"] for r in rows),
                  baseline_total_seconds=baseline_total, component_total_seconds=component_total,
                  total_speedup=baseline_total / component_total,
                  elapsed_seconds=time.perf_counter() - started,
                  caveat="Component independence guarantees cardinality and total-distance objectives; tied optimal assignments may change distance distribution/p95.",
                  rows=rows)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"PROBE COMPLETE {output}; speedup={report['total_speedup']:.3f}; "
          f"all sorted distances equal={report['all_sorted_distances_equal']}", flush=True)


if __name__ == "__main__":
    main()
