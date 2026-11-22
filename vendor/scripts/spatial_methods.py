"""Classic spatial hysteresis and minimum-component-size edge selection.

``keep`` must be the original gradient-magnitude NMS mask, and ``score`` the
same-scale noise-normalized gradient confidence C.  This module does not
estimate noise, compute gradients, or read images, labels, dataset splits,
or optimization results.  It is ordinary connected-component postprocessing,
not a new edge-detection mechanism.

All connectivity and component sizes are computed on the WHOLE image using
eight-neighbour connectivity.  A component touching the frame is neither
discarded nor truncated here.  The shared evaluation code should exclude its
common 12-pixel border only AFTER detection; cropping before propagation
would change the algorithm.  ``min_size`` counts component pixels, not an
estimated arc length.  No morphology/thinning or gap bridging is applied.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


_EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=bool)


def connected_edge(keep, score, high, low_ratio=.6, min_size=0,
                   hysteresis=True) -> np.ndarray:
    """Return a new boolean edge map of the input image's original shape.

    With hysteresis, weak pixels satisfy ``keep & (score >= high*low_ratio)``.
    Retain an entire eight-connected weak component iff it contains at least
    one strong pixel satisfying ``keep & (score >= high)``.  Then retain only
    components with at least ``min_size`` pixels.  A low_ratio of one makes
    weak and strong thresholds identical.

    Without hysteresis, first take the original single-threshold NMS pixels
    ``keep & (score >= high)`` and optionally filter their connected sizes.
    In particular, ``hysteresis=False, min_size=0`` is exactly the original
    fixed-scale decision.  A size of zero means no size-based filtering.
    Neither input is modified.  Equality at all thresholds is retained.
    """
    nms_mask = np.asarray(keep, dtype=bool)
    confidence = np.asarray(score)
    if nms_mask.ndim != 2 or confidence.shape != nms_mask.shape:
        raise ValueError("keep and score must have the same two-dimensional shape")
    if (not np.issubdtype(confidence.dtype, np.number)
            or np.issubdtype(confidence.dtype, np.complexfloating)
            or not np.isfinite(confidence).all()):
        raise ValueError("score must contain finite real numbers")
    high = float(high)
    low_ratio = float(low_ratio)
    if not np.isfinite(high) or high <= 0:
        raise ValueError("high must be finite and positive")
    if not np.isfinite(low_ratio) or not 0 <= low_ratio <= 1:
        raise ValueError("low_ratio must be in [0, 1]")
    if (not isinstance(min_size, (int, np.integer)) or isinstance(min_size, bool)
            or min_size < 0):
        raise ValueError("min_size must be a nonnegative integer")
    strong = nms_mask & (confidence >= high)
    if not np.any(strong):
        return np.zeros(nms_mask.shape, dtype=bool)
    if not hysteresis or low_ratio == 1.0:
        candidates = strong
        if min_size <= 1:
            return strong
    else:
        candidates = nms_mask & (confidence >= high * low_ratio)
    labels, count = ndi.label(candidates, structure=_EIGHT_NEIGHBOURS)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    contains_strong = np.bincount(labels[strong], minlength=count + 1) > 0
    retained = contains_strong & (sizes >= min_size)
    retained[0] = False
    return retained[labels]


def _reference(keep, score, high, low_ratio, min_size, hysteresis):
    """Independent small-array flood-fill oracle, used only by the self-check."""
    from collections import deque

    strong = keep & (score >= high)
    candidates = keep & (score >= high * low_ratio) if hysteresis else strong
    visited = np.zeros_like(keep, dtype=bool)
    output = np.zeros_like(keep, dtype=bool)
    height, width = keep.shape
    for first_y, first_x in np.argwhere(candidates):
        if visited[first_y, first_x]:
            continue
        queue = deque([(int(first_y), int(first_x))])
        visited[first_y, first_x] = True
        component = []
        has_strong = False
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            has_strong |= bool(strong[y, x])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if ((dy or dx) and 0 <= ny < height and 0 <= nx < width
                            and candidates[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if has_strong and len(component) >= min_size:
            for y, x in component:
                output[y, x] = True
    return output


def _verify():
    """Meaningful small-array checks; no image dataset is accessed."""
    import hashlib
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    checks = {}
    score = np.zeros((8, 12), dtype=np.float32)
    # Strong seed followed by weak diagonal/axis-connected pixels.
    score[1, 1] = 10
    score[2, 2] = 6
    score[3, 2] = 7
    # Weak component with no strong seed.
    score[1, 9] = score[2, 9] = 8
    # One isolated strong seed, removed by min_size > 1.
    score[6, 9] = 11
    keep = score > 0
    keep_before, score_before = keep.copy(), score.copy()
    edge = connected_edge(keep, score, high=10, low_ratio=.6)
    checks["seeded_weak_segment_retained"] = bool(edge[1, 1] and edge[2, 2] and edge[3, 2])
    checks["diagonal_8_connectivity"] = bool(edge[2, 2])
    checks["unseeded_weak_segment_discarded"] = not bool(edge[1, 9] or edge[2, 9])
    filtered = connected_edge(keep, score, 10, .6, 3)
    checks["min_size_counts_weak_component_after_hysteresis"] = bool(
        filtered.sum() == 3 and filtered[1, 1] and not filtered[6, 9])
    checks["component_at_size_limit_retained"] = bool(filtered.sum() == 3)
    checks["above_size_limit_discarded"] = not bool(connected_edge(keep, score, 10, .6, 4).any())
    direct = connected_edge(keep, score, 10, min_size=0, hysteresis=False)
    checks["no_hysteresis_min0_exact_original_fixed"] = bool(np.array_equal(direct, keep & (score >= 10)))
    checks["equal_thresholds_equal_direct"] = bool(np.array_equal(connected_edge(keep, score, 10, 1), direct))
    checks["inputs_unchanged"] = bool(np.array_equal(keep, keep_before) and np.array_equal(score, score_before))
    blocked = keep.copy()
    blocked[2, 2] = False
    blocked_output = connected_edge(blocked, score, 10, .6)
    checks["non_nms_pixel_never_bridges_components"] = not bool(blocked_output[3, 2])
    # A frame seed may connect to an interior weak segment; no crop here.
    border_score = np.zeros((5, 6), dtype=np.float32)
    border_score[0, 0] = 10
    border_score[1, 1] = border_score[2, 2] = 6
    checks["frame_contact_preserved_before_external_score_mask"] = bool(
        connected_edge(border_score > 0, border_score, 10, .6).sum() == 3)
    no_seed = np.ones((3, 3), dtype=np.float32)
    checks["all_weak_without_seed_empty"] = not bool(connected_edge(no_seed > 0, no_seed, 10, .1).any())
    checks["empty_mask_empty"] = not bool(connected_edge(np.zeros_like(keep), score, 10).any())
    checks["empty_2d_input_supported"] = connected_edge(np.empty((0, 4), bool), np.empty((0, 4)), 10).shape == (0, 4)
    checks["boolean_original_shape"] = bool(edge.dtype == np.bool_ and edge.shape == keep.shape)
    rng = np.random.default_rng(5_351_031)
    reference_comparisons = 0
    for _ in range(40):
        random_keep = rng.random((7, 9)) > .35
        random_score = rng.integers(0, 15, (7, 9)).astype(np.float32)
        for ratio in (0.0, .3, .6, 1.0):
            for size in (0, 4, 8):
                for use_hysteresis in (False, True):
                    observed = connected_edge(random_keep, random_score, 10, ratio, size, use_hysteresis)
                    expected = _reference(random_keep, random_score, 10, ratio, size, use_hysteresis)
                    assert np.array_equal(observed, expected)
                    reference_comparisons += 1
    checks["independent_flood_fill_oracle"] = True
    assert all(checks.values()), checks
    source = Path(__file__).resolve()
    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "passed": True,
        "checks": checks,
        "independent_random_small_array_comparisons": reference_comparisons,
        "random_seed": 5_351_031,
        "dataset_accessed": False,
        "connectivity": 8,
        "filter_order": "NMS/threshold -> seeded weak components -> min component pixel count",
        "boundary_policy": "whole-image connectivity; external evaluator applies common border12 afterward",
        "min_size_definition": "component pixel count, not arc length",
    }
    destination = source.parents[1] / "audit" / "spatial_verification.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Spatial verification passed: {len(checks)} behavior checks; "
          f"{reference_comparisons} independent reference comparisons; {destination}")


if __name__ == "__main__":
    _verify()
