"""Fresh v7 seed namespaces with unchanged sixth-round image definitions.

No external images or BSDS files are read. Only the observed ``input`` is a
detector input. Clean, truth, and nominal sigma arrays are evaluation data.
The self-check renders calibration only; locked seeds are enumerated only.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re

import numpy as np
from data_v6 import (Config, SIZE, CONTRASTS, SIGMAS, FAMILIES, SHAPES,
                     _geometry, _noise, _boundary, _specs as _v6_specs)

NAMESPACES = {'calibration': 'cal-v7', 'development': 'dev-v7',
              'locked_test': 'test-v7', 'locked_null': 'null-v7'}


def _seed(split, shape, contrast, family, rep, version=7):
    if version not in (5, 6, 7):
        raise ValueError('supported seed versions are 5, 6, and 7')
    suffix = 'v1' if version == 5 else f'v{version}'
    prefix = {'calibration': 'cal', 'development': 'dev',
              'locked_test': 'test', 'locked_null': 'null'}[split]
    key = f'{prefix}-{suffix}:{shape}:{contrast:g}:{family}:{rep}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def _specs(split, version=7):
    """Enumerate the fixed design; never render a locked image."""
    if split not in NAMESPACES:
        raise ValueError(split)
    if version not in (5, 6, 7):
        raise ValueError('supported design versions are 5, 6, and 7')
    yield from _v6_specs(split, version=5 if version == 5 else 6)


def generate_cases(split='development', frozen_protocol_sha256=None, config=None):
    """Render fixed v7 design; locked splits require a 64-character hex hash."""
    if split not in NAMESPACES:
        raise ValueError(split)
    if split.startswith('locked_') and not (
            isinstance(frozen_protocol_sha256, str)
            and re.fullmatch(r'[0-9a-fA-F]{64}', frozen_protocol_sha256)):
        raise ValueError('locked split requires a 64-hex frozen_protocol_sha256')
    cfg = config or Config()
    if cfg.size != SIZE:
        raise ValueError('round-seven geometry protocol fixes size=128')
    out = []
    for shape, contrast, family, sigma, rep, empty in _specs(split):
        seed = _seed(split, shape, contrast, family, rep)
        clean, truth, geometry = _geometry(shape, contrast, seed, cfg, empty)
        image, residual, sigma_map = _noise(clean, sigma, family,
                                            seed ^ 0x9E3779B97F4A7C15, cfg)
        out.append(dict(input=image, clean=clean, truth=truth,
                        true_sigma=sigma_map, noise_residual=residual,
                        metadata=dict(
                            case_id=f'v7_{split}_{shape}_{contrast:g}_{family}_{rep}',
                            scene=shape, contrast=contrast, noise_family=family,
                            sigma=sigma, seed=seed, split=split,
                            seed_namespace=NAMESPACES[split],
                            frozen_protocol_sha256=frozen_protocol_sha256,
                            geometry=geometry, geometry_empty=empty,
                            gt_boundary_pixels=int(truth.sum()),
                            background=dict(level=cfg.background_level,
                                            slope_x=cfg.slope_x, slope_y=cfg.slope_y),
                            truth_rule='non-wrapping inner four-neighbour boundary of binary geometry mask',
                            noise_sensor_pattern='banding is additive sensor interference and never GT',
                            true_sigma_semantics='nominal additive random-noise scale; only spatial_white varies by pixel; not covariance or JPEG/quantization error variance',
                            quantization_step=cfg.quant_step, jpeg_quality=cfg.jpeg_quality,
                            clipping_policy='JPEG clips to [0,255] before uint8 encoding; other families not clipped',
                            design='shape*contrast*noise_family factorial; sigma Latin-square marginal balance, not sigma full factorial')))
    return out


def self_check():
    """Render cal only; enumerate all versions' seeds without locked rendering."""
    cal = generate_cases('calibration')
    repeated = generate_cases('calibration')
    seeds = {split: {_seed(split, a, b, c, e) for a, b, c, d, e, f in _specs(split)}
             for split in NAMESPACES}
    historical = {version: {_seed(split, a, b, c, e, version=version)
                           for split in NAMESPACES
                           for a, b, c, d, e, f in _specs(split, version=version)}
                  for version in (5, 6)}
    for i, split in enumerate(NAMESPACES):
        assert len(seeds[split]) == len(list(_specs(split)))
        for prior in historical.values():
            assert seeds[split].isdisjoint(prior)
        for other in list(NAMESPACES)[i+1:]:
            assert seeds[split].isdisjoint(seeds[other])
    assert len(cal) == 96 and all(not c['truth'].any() for c in cal)
    assert all(np.isfinite(c[k]).all() for c in cal
               for k in ('input', 'clean', 'true_sigma', 'noise_residual'))
    assert all(np.array_equal(a['input'], b['input'])
               and np.array_equal(a['truth'], b['truth'])
               for a, b in zip(cal, repeated))
    assert set(Counter(c['metadata']['noise_family'] for c in cal).values()) == {16}
    assert set(Counter(c['metadata']['sigma'] for c in cal).values()) == {24}
    sigma_counts = {split: dict(Counter(spec[3] for spec in _specs(split)))
                    for split in NAMESPACES}
    assert all(len(set(counts.values())) == 1 for counts in sigma_counts.values())
    for split in ('locked_test', 'locked_null'):
        for bad in (None, '', 'x'*64, 'a'*63, 'a'*65, 123):
            try:
                generate_cases(split, frozen_protocol_sha256=bad)
            except ValueError:
                pass
            else:
                raise AssertionError('malformed hash was accepted')
    frame = np.zeros((5, 5), bool); frame[:, 0] = True
    assert np.array_equal(_boundary(frame), frame)
    source = Path(__file__).resolve()
    definition_source = source.parent/'data_v6.py'
    report = dict(passed=True,
                  counts={split: len(list(_specs(split))) for split in NAMESPACES},
                  rendered_splits=['calibration'], locked_splits_not_rendered=True,
                  development_not_rendered=True,
                  seed_namespaces=NAMESPACES, pairwise_seed_disjoint=True,
                  disjoint_from_v5_all_splits=True, disjoint_from_v6_all_splits=True,
                  cal_family_counts=dict(Counter(c['metadata']['noise_family'] for c in cal)),
                  split_sigma_counts=sigma_counts,
                  deterministic_calibration_inputs_and_truth=True,
                  malformed_locked_hashes_rejected=True,
                  nonwrapping_boundary_verified=True,
                  sigma_design='marginal Latin-square, not full sigma factorial',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  image_definition_dependency_sha256=hashlib.sha256(definition_source.read_bytes()).hexdigest(),
                  cal_input_sha256=hashlib.sha256(b''.join(c['input'].tobytes() for c in cal)).hexdigest(),
                  note='All six families participate in calibration; no OOD claim. Development is available but not used in v7 parameter selection.')
    output = source.parents[1]/'audit'/'data_v7_verification.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(self_check(), ensure_ascii=False, indent=2))
