"""Scale-specific empirical gradient-noise normalization.

The classroom estimator obtains a pixel-noise variance from a 2x2 high-pass
residual and propagates it through squared derivative kernels. That propagation
assumes uncorrelated pixel noise and can greatly underestimate smooth correlated
noise. This module supplements it with a robust empirical trace measured on
the actual horizontal and vertical derivative images at EACH scale.

No clean image, annotation, noise label, noise parameter, or dataset is accepted.
The gradient-domain MAD is an image statistic, not an uncertainty confidence
interval or a covariance model. Texture, edges, and nonstationary interference
can inflate it and reduce recall; nonlinear processing can violate its Gaussian
interpretation. Detection thresholds still require development/calibration.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from optimization_bank import OptimizationBank
from bsds_support import core


class RobustBank(OptimizationBank):
    """OptimizationBank-compatible gradient bank with an empirical noise floor.

    ``RobustBank(image, noise_mode='robust')`` is directly accepted by the
    fifth-round ContourEvidence class because this class inherits its bank API.
    Numeric second arguments to get_bank remain lambda, as in OptimizationBank.
    A mode can be specified with get_bank(scale, mode='original'|'robust').
    The convenience get_bank(scale, 'original') is supported as well.
    """

    MAD_NORMALIZER = 0.67449

    def __init__(self, image, noise_mode="robust"):
        if noise_mode not in ("original", "robust"):
            raise ValueError("noise_mode must be 'original' or 'robust'")
        super().__init__(image)
        if min(self.shape) < 2:
            raise ValueError("both image dimensions must have at least two pixels")
        self.noise_mode = noise_mode
        self._empirical = {}
        self._robust_banks = {}
        self._diagnostics = {}

    def _empirical_noise(self, scale):
        scale = float(scale)
        if scale not in self._empirical:
            hx, hy = core.derivative_kernels(scale)
            gx = ndi.convolve(self.image.astype(np.float64), hx, mode="reflect")
            gy = ndi.convolve(self.image.astype(np.float64), hy, mode="reflect")
            y_starts = list(range(0, self.shape[0], self.TILE))
            x_starts = list(range(0, self.shape[1], self.TILE))
            centers_y = [(y + min(y + self.TILE, self.shape[0]) - 1) / 2 for y in y_starts]
            centers_x = [(x + min(x + self.TILE, self.shape[1]) - 1) / 2 for x in x_starts]
            trace_grid = np.empty((len(y_starts), len(x_starts)), dtype=np.float64)
            for iy, y in enumerate(y_starts):
                for ix, x in enumerate(x_starts):
                    components = [g[y:min(y + self.TILE, self.shape[0]),
                                    x:min(x + self.TILE, self.shape[1])]
                                  for g in (gx, gy)]
                    trace_grid[iy, ix] = sum(
                        (np.median(np.abs(block - np.median(block))) /
                         self.MAD_NORMALIZER) ** 2 for block in components)
            # Use the same actual tile centres and nearest-edge extrapolation
            # as the original classroom pixel-variance interpolation.
            trace = core._interpolate_grid(trace_grid, centers_y, centers_x,
                                           self.shape).astype(np.float64)
            self._empirical[scale] = dict(trace=trace, trace_grid=trace_grid)
        return self._empirical[scale]

    def get_bank(self, scale, lam=0, mode=None):
        if isinstance(lam, str):
            if mode is not None:
                raise ValueError("mode supplied twice")
            mode, lam = lam, 0
        mode = self.noise_mode if mode is None else mode
        if mode not in ("original", "robust"):
            raise ValueError("mode must be 'original' or 'robust'")
        scale, lam = float(scale), float(lam)
        original = super().get_bank(scale, lam)
        if mode == "original":
            return original
        key = (scale, lam)
        if key not in self._robust_banks:
            empirical = self._empirical_noise(scale)
            # Original N2 is the same public float32 variance returned by the
            # base bank. Rounding in C must never increase the original score.
            noise2 = np.maximum(original["N2"].astype(np.float64), empirical["trace"])
            confidence = self._gradient(scale)["magnitude64"] / np.sqrt(noise2 + self.EPSILON)
            confidence = np.minimum(confidence.astype(np.float32), original["C"])
            self._robust_banks[key] = dict(M=original["M"], theta=original["theta"],
                                           C=confidence, N2=noise2.astype(np.float32),
                                           keep=original["keep"])
            inner = np.s_[max(1, int(np.ceil(3 * scale))):-max(1, int(np.ceil(3 * scale))),
                          max(1, int(np.ceil(3 * scale))):-max(1, int(np.ceil(3 * scale)))]
            reference = original["N2"][inner].astype(np.float64)
            trace = empirical["trace"][inner]
            adjusted = noise2[inner]
            if reference.size == 0:
                reference, trace, adjusted = (original["N2"].astype(np.float64),
                                               empirical["trace"], noise2)
            ratio = adjusted / np.maximum(reference, self.EPSILON)
            self._diagnostics[key] = dict(
                scale=scale, lam=lam, mode="robust", tile=self.TILE,
                original_noise2_median=float(np.median(reference)),
                empirical_trace_median=float(np.median(trace)),
                robust_noise2_median=float(np.median(adjusted)),
                noise2_ratio_median=float(np.median(ratio)),
                noise2_ratio_p95=float(np.quantile(ratio, .95)),
                fraction_floor_active=float(np.mean(trace > reference)),
                formula="N2=max(N2_classroom,MAD(gx)^2/.67449^2+MAD(gy)^2/.67449^2)",
                caveat="Gradient-domain robust image statistic; structure and spatial dependence can bias it. Not a confidence interval.")
        return self._robust_banks[key]

    def diagnostics(self, scale, lam=0):
        """Return JSON-compatible scale-specific normalization diagnostics."""
        self.get_bank(scale, lam=lam, mode="robust")
        return dict(self._diagnostics[(float(scale), float(lam))])


def _self_check():
    """Generated-noise implementation checks; no experiment split is read."""
    from pathlib import Path
    import hashlib
    import json
    import time

    started = time.perf_counter()
    rng = np.random.default_rng(6_110_101)
    independent = rng.normal(0, .02, (192, 192)).astype(np.float32)
    smooth = ndi.gaussian_filter(rng.normal(size=independent.shape), 1.25)
    correlated = (smooth * (.02 / smooth.std())).astype(np.float32)
    checks = {}
    ratios = []
    for noise_name, image in (("independent", independent), ("correlated", correlated)):
        source = image.copy()
        base, new = OptimizationBank(image), RobustBank(image)
        for scale in new.SCALES:
            original = base.get_bank(scale, 0)
            public_original = new.get_bank(scale, mode="original")
            adjusted = new.get_bank(scale, mode="robust")
            assert all(np.array_equal(original[k], public_original[k]) for k in original)
            assert all(np.isfinite(adjusted[k]).all() for k in ("M", "theta", "C", "N2"))
            assert np.all(adjusted["C"] <= original["C"])
            assert np.all(adjusted["N2"] >= original["N2"])
            assert all(np.array_equal(adjusted[k], original[k]) for k in ("M", "theta", "keep"))
            hx, hy = core.derivative_kernels(scale)
            gx = ndi.convolve(image.astype(np.float64), hx, mode="reflect")
            gy = ndi.convolve(image.astype(np.float64), hy, mode="reflect")
            border = max(10, int(np.ceil(3 * scale)))
            inner = np.s_[border:-border, border:-border]
            actual_trace = float(np.var(gx[inner]) + np.var(gy[inner]))
            raw_ratio = float(np.median(original["N2"][inner]) / actual_trace)
            robust_ratio = float(np.median(adjusted["N2"][inner]) / actual_trace)
            ratios.append(dict(noise=noise_name, scale=scale, actual_gradient_trace=actual_trace,
                               original_over_actual=raw_ratio, robust_over_actual=robust_ratio,
                               diagnostics=new.diagnostics(scale)))
        assert np.array_equal(source, image)
    checks["original_mode_exact_array_equality"] = True
    checks["finite_scores_and_variances"] = True
    checks["robust_confidence_never_above_original"] = True
    checks["robust_variance_never_below_original"] = True
    checks["gradient_direction_nms_unchanged"] = True
    checks["input_unmodified"] = True
    checks["constant_image_no_candidate"] = not RobustBank(np.full((64, 64), .18, np.float32)).get_bank(1.4)["keep"].any()
    checks["independent_noise_trace_roughly_correct"] = all(.65 < r["robust_over_actual"] < 1.5 for r in ratios if r["noise"] == "independent")
    checks["correlated_noise_trace_improves"] = all(abs(np.log(r["robust_over_actual"])) < abs(np.log(r["original_over_actual"])) for r in ratios if r["noise"] == "correlated")
    checks["correlated_noise_trace_roughly_correct"] = all(.5 < r["robust_over_actual"] < 1.8 for r in ratios if r["noise"] == "correlated")
    try:
        RobustBank(independent, noise_mode="unrecognized")
    except ValueError:
        checks["invalid_mode_rejected"] = True
    assert all(checks.values()), checks
    source = Path(__file__).resolve()
    report = dict(passed=True, checks=checks, noise_trace_ratios=ratios,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  experiment_split_accessed=False, elapsed_seconds=time.perf_counter()-started)
    out = source.parents[1] / "audit" / "robust_bank_verification.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _self_check()
