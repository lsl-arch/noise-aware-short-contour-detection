"""Independent-seed synthetic data for the sixth-round edge experiment.

Geometry and corruption definitions intentionally match round five.  Every
split has a new v6 seed namespace.  Calibration now covers all six corruption
families, so these families are in-distribution in round six, not OOD.
Only ``input`` may be passed to a detector; the other arrays are evaluation
references.  The nominal sigma map is not a JPEG/quantization error variance.
"""
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image

SIZE = 128
CONTRASTS = (.01, .02, .04, .08)
SIGMAS = (.01, .02, .03, .04)
FAMILIES = ('uniform_white', 'spatial_white', 'correlated', 'banding', 'quantization', 'jpeg')
SHAPES = ('shortbar', 'smallcircle', 'shortL', 'blurredsemidisc')
NAMESPACES = {'calibration': 'cal-v6', 'development': 'dev-v6',
              'locked_test': 'test-v6', 'locked_null': 'null-v6'}


@dataclass(frozen=True)
class Config:
    size: int = SIZE
    background_level: float = .18
    slope_x: float = .025
    slope_y: float = -.018
    blur_sigma: float = 1.15
    quant_step: float = 1 / 255
    jpeg_quality: int = 38


def _seed(split, shape, contrast, family, rep, version=6):
    namespace = NAMESPACES[split] if version == 6 else {
        'calibration': 'cal-v1', 'development': 'dev-v1',
        'locked_test': 'test-v1', 'locked_null': 'null-v1'}[split]
    key = f'{namespace}:{shape}:{contrast:g}:{family}:{rep}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def _specs(split, version=6):
    """Enumerate metadata only; this never renders a locked image."""
    if split not in NAMESPACES:
        raise ValueError(split)
    if split == 'calibration':
        if version == 5:
            for i in range(96):
                family = 'uniform_white' if i < 48 else 'spatial_white'
                yield 'none', 0., family, SIGMAS[i % 4], i, True
        else:
            for family_index, family in enumerate(FAMILIES):
                for rep in range(16):
                    yield 'none', 0., family, SIGMAS[(family_index + rep) % 4], rep, True
    elif split == 'locked_null':
        reps = 16 if version == 5 else 32
        for family_index, family in enumerate(FAMILIES):
            for rep in range(reps):
                # Round-five used a global rep id for null images.
                rep_id = family_index * 16 + rep if version == 5 else rep
                yield 'none', 0., family, SIGMAS[(family_index + rep) % 4], rep_id, True
    else:
        reps = 1 if split == 'development' else 2
        for si, shape in enumerate(SHAPES):
            for ci, contrast in enumerate(CONTRASTS):
                for fi, family in enumerate(FAMILIES):
                    for rep in range(reps):
                        yield shape, contrast, family, SIGMAS[(si * 4 + ci + fi + rep) % 4], rep, False


def _boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                            & mask[2:, 1:-1] & mask[1:-1, :-2]
                            & mask[1:-1, 2:])
    return mask & ~interior


def _geometry(shape, contrast, seed, cfg, empty=False):
    n = cfg.size
    y, x = np.indices((n, n))
    rng = np.random.default_rng(seed)
    base = (np.full((n, n), cfg.background_level, np.float32)
            + cfg.slope_x * (x - n / 2) / n + cfg.slope_y * (y - n / 2) / n)
    mask = np.zeros((n, n), bool)
    params = {}
    if empty:
        return base, mask, params
    cy, cx = int(rng.integers(45, 83)), int(rng.integers(45, 83))
    if shape == 'shortbar':
        length, width = int(rng.integers(6, 19)), int(rng.integers(3, 8))
        theta = float(rng.uniform(0, np.pi))
        t = np.cos(theta) * (x - cx) + np.sin(theta) * (y - cy)
        q = -np.sin(theta) * (x - cx) + np.cos(theta) * (y - cy)
        mask = (abs(t) <= length / 2) & (abs(q) <= width / 2)
        params.update(length=length, width=width, theta=theta)
    elif shape == 'smallcircle':
        radius = int(rng.integers(3, 10))
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        params.update(radius=radius)
    elif shape == 'shortL':
        length, width = int(rng.integers(7, 19)), int(rng.integers(3, 6))
        a = (abs(x - cx) <= width) & (y >= cy) & (y <= cy + length)
        b = (abs(y - (cy + length)) <= width) & (x >= cx) & (x <= cx + length)
        mask = a | b
        params.update(length=length, width=width)
    elif shape == 'blurredsemidisc':
        radius = int(rng.integers(5, 10))
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2) & (y <= cy)
        params.update(radius=radius, blur_sigma=cfg.blur_sigma)
    else:
        raise ValueError(shape)
    if shape == 'blurredsemidisc':
        from scipy.ndimage import gaussian_filter
        rendered = gaussian_filter(mask.astype(np.float32), cfg.blur_sigma, mode='reflect')
    else:
        rendered = mask.astype(np.float32)
    clean = base + contrast * rendered
    params.update(center_y=cy, center_x=cx)
    return clean, _boundary(mask), params


def _noise(clean, sigma, family, seed, cfg):
    rng = np.random.default_rng(seed)
    level = np.clip(clean, 0, 1)
    local = sigma * (.65 + .7 * (level - level.min()) / (np.ptp(level) + 1e-12))
    z = rng.standard_normal(clean.shape)
    sigma_out = np.full(clean.shape, sigma, np.float32)
    if family == 'uniform_white':
        residual = z * sigma
    elif family == 'spatial_white':
        residual, sigma_out = z * local, local
    elif family == 'correlated':
        from scipy.ndimage import gaussian_filter
        filtered = gaussian_filter(z, 1.25)
        residual = filtered * sigma / max(float(filtered.std()), 1e-8)
    elif family == 'banding':
        residual = z * sigma + np.sin(np.arange(clean.shape[1])[None, :] * 2 * np.pi / 11) * sigma * .9
    elif family == 'quantization':
        residual = np.round((clean + z * sigma) / cfg.quant_step) * cfg.quant_step - clean
    elif family == 'jpeg':
        arr = np.uint8(np.clip((clean + z * sigma) * 255, 0, 255))
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=cfg.jpeg_quality)
        out = np.asarray(Image.open(io.BytesIO(buf.getvalue())), np.float32) / 255
        residual = out - clean
    else:
        raise ValueError(family)
    return (clean + residual).astype(np.float32), residual.astype(np.float32), sigma_out.astype(np.float32)


def generate_cases(split='development', frozen_protocol_sha256=None, config=None):
    """Return arrays and metadata; locked rendering requires a 64-hex hash."""
    if split not in NAMESPACES:
        raise ValueError(split)
    if split.startswith('locked_') and not (isinstance(frozen_protocol_sha256, str)
                                             and re.fullmatch(r'[0-9a-fA-F]{64}', frozen_protocol_sha256)):
        raise ValueError('locked split requires a 64-hex frozen_protocol_sha256')
    cfg = config or Config()
    if cfg.size != SIZE:
        raise ValueError('round-six geometry protocol fixes size=128')
    out = []
    for shape, contrast, family, sigma, rep, empty in _specs(split):
        seed = _seed(split, shape, contrast, family, rep)
        clean, truth, geometry = _geometry(shape, contrast, seed, cfg, empty)
        image, residual, sigma_map = _noise(clean, sigma, family, seed ^ 0x9E3779B97F4A7C15, cfg)
        case_id = f'v6_{split}_{shape}_{contrast:g}_{family}_{rep}'
        out.append(dict(input=image, clean=clean, truth=truth, true_sigma=sigma_map,
                        noise_residual=residual, metadata=dict(
                            case_id=case_id, scene=shape, contrast=contrast,
                            noise_family=family, sigma=sigma, seed=seed, split=split,
                            seed_namespace=NAMESPACES[split], frozen_protocol_sha256=frozen_protocol_sha256,
                            geometry=geometry, geometry_empty=empty,
                            gt_boundary_pixels=int(truth.sum()),
                            background=dict(level=cfg.background_level, slope_x=cfg.slope_x, slope_y=cfg.slope_y),
                            truth_rule='non-wrapping inner four-neighbour boundary of binary geometry mask',
                            noise_sensor_pattern='banding is additive sensor interference and never GT',
                            true_sigma_semantics='nominal additive random-noise scale; only spatial_white varies by pixel; not covariance or JPEG/quantization error variance',
                            quantization_step=cfg.quant_step, jpeg_quality=cfg.jpeg_quality,
                            clipping_policy='JPEG clips to [0,255] before uint8 encoding; other families not clipped',
                            design='shape*contrast*noise_family factorial; sigma Latin-square marginal balance, not sigma full factorial')))
    return out


def self_check():
    """Render cal/dev only; enumerate locked specs without creating images."""
    dev, cal = generate_cases('development'), generate_cases('calibration')
    seeds = {split: {_seed(split, a, b, c, e) for a, b, c, d, e, f in _specs(split)}
             for split in NAMESPACES}
    old_seeds = {_seed(split, a, b, c, e, version=5) for split in NAMESPACES
                 for a, b, c, d, e, f in _specs(split, version=5)}
    for i, split in enumerate(NAMESPACES):
        assert len(seeds[split]) == len(list(_specs(split)))
        assert seeds[split].isdisjoint(old_seeds)
        for other in list(NAMESPACES)[i + 1:]:
            assert seeds[split].isdisjoint(seeds[other])
    assert len(dev) == 96 and len(cal) == 96
    assert all(c['truth'].any() for c in dev)
    assert all(not c['truth'].any() for c in cal)
    assert all(np.isfinite(c[key]).all() for c in dev + cal for key in ('input', 'clean', 'true_sigma', 'noise_residual'))
    repeated = generate_cases('development')
    assert all(np.array_equal(a['input'], b['input']) and np.array_equal(a['truth'], b['truth'])
               for a, b in zip(dev, repeated))
    assert set(Counter(c['metadata']['noise_family'] for c in cal).values()) == {16}
    assert set(Counter(c['metadata']['sigma'] for c in cal).values()) == {24}
    assert set(Counter(c['metadata']['sigma'] for c in dev).values()) == {24}
    for split in ('locked_test', 'locked_null'):
        for bad in (None, '', 'x' * 64, 'a' * 63, 123):
            try:
                generate_cases(split, frozen_protocol_sha256=bad)
            except ValueError:
                pass
            else:
                raise AssertionError('malformed hash was accepted')
    frame = np.zeros((5, 5), bool)
    frame[:, 0] = True
    assert np.array_equal(_boundary(frame), frame)
    assert all(not c['truth'][[0, -1], :].any() and not c['truth'][:, [0, -1]].any() for c in dev)
    report = dict(passed=True, counts={split: len(list(_specs(split))) for split in NAMESPACES},
                  rendered_splits=['calibration', 'development'], locked_splits_not_rendered=True,
                  seed_namespaces=NAMESPACES, pairwise_seed_disjoint=True, disjoint_from_v5_all_splits=True,
                  cal_family_counts=dict(Counter(c['metadata']['noise_family'] for c in cal)),
                  cal_sigma_counts=dict(Counter(c['metadata']['sigma'] for c in cal)),
                  dev_sigma_counts=dict(Counter(c['metadata']['sigma'] for c in dev)),
                  deterministic_inputs_and_truth=True, malformed_locked_hashes_rejected=True,
                  nonwrapping_boundary_verified=True, sigma_design='marginal Latin-square, not full sigma factorial',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='all six noise families now participate in calibration; no OOD claim for those six',
                  dev_input_sha256=hashlib.sha256(b''.join(c['input'].tobytes() for c in dev)).hexdigest())
    path = Path(__file__).resolve().parents[1] / 'audit' / 'data_verification.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(self_check(), ensure_ascii=False, indent=2))
