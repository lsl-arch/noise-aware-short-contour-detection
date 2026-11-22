"""Image-only refinements of the scale-specific gradient-noise floor.

The previous MAD32 estimator may count real texture/edges as noise. For each
32-pixel tile, this module can instead compute four MAD traces on its 16-pixel
quadrants and combine them using the lower quartile or median. Truncated tiles
split their actual samples into at most four nonempty quadrants. The resulting
tile statistic is interpolated at the original actual tile centres.

These are empirical image statistics, NOT unbiased noise estimates, covariance
models, probabilities or confidence intervals. Selecting a low sub-tile
quantile can underestimate noise even on a homogeneous background; alpha<1
relaxes the empirical floor further. Texture may still contaminate every
subtile. Calibration/development must therefore assess the rejection/recall
tradeoff, and no statistical error guarantee is provided.

Only the observed 2-D image enters the class. It does not accept truth, clean
images, true noise, noise-family IDs, dataset paths or externally fitted maps.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from bsds_support import core
from robust_bank import RobustBank


class RefinedBank(RobustBank):
    """Reusable bank for MAD32, lower-quartile and median sub-tile floors.

    ``bank = RefinedBank(image, strategy='sub_q25', alpha=.5)``
    ``bank.get_bank(scale, lam=0, strategy=None, alpha=None)``

    Optional strategy/alpha overrides make one image's gradients and tile
    statistics reusable across a finite development grid. ``alpha=0`` returns
    the exact original bank, and ``mad32,alpha=1`` returns the exact previous
    RobustBank. Gradient magnitudes, directions, NMS and pixel V/D are unchanged.
    """

    STRATEGIES = ("mad32", "sub_q25", "sub_median")

    def __init__(self, image, strategy="mad32", alpha=1.0):
        self._validate(strategy, alpha)
        super().__init__(image, noise_mode="robust")
        self.strategy = strategy
        self.alpha = float(alpha)
        self._subtiles = {}
        self._refined_banks = {}
        self._refined_diagnostics = {}

    @classmethod
    def _validate(cls, strategy, alpha):
        if strategy not in cls.STRATEGIES:
            raise ValueError("strategy must be mad32, sub_q25 or sub_median")
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0,1]")

    def _subtile_fields(self, scale):
        scale = float(scale)
        if scale not in self._subtiles:
            hx, hy = core.derivative_kernels(scale)
            gradients = [ndi.convolve(self.image.astype(np.float64), kernel,
                                      mode="reflect") for kernel in (hx, hy)]
            ys = list(range(0, self.shape[0], self.TILE))
            xs = list(range(0, self.shape[1], self.TILE))
            centers_y = [(y + min(y+self.TILE, self.shape[0])-1)/2 for y in ys]
            centers_x = [(x + min(x+self.TILE, self.shape[1])-1)/2 for x in xs]
            grids = {key: np.empty((len(ys), len(xs)), dtype=np.float64)
                     for key in ("sub_q25", "sub_median")}
            counts = np.zeros((len(ys), len(xs)), dtype=np.int16)
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    blocks = [g[y:min(y+self.TILE, self.shape[0]),
                                x:min(x+self.TILE, self.shape[1])]
                              for g in gradients]
                    # Split actual tiles approximately equally, just as the
                    # classroom V/D estimator does at truncated image borders.
                    subtraces = []
                    rows = [np.array_split(block, 2, axis=0) for block in blocks]
                    for ry in range(2):
                        cols = [np.array_split(part[ry], 2, axis=1) for part in rows]
                        for cx in range(2):
                            components = [part[cx] for part in cols]
                            if not components[0].size:
                                continue
                            trace = sum((np.median(np.abs(block-np.median(block))) /
                                         self.MAD_NORMALIZER)**2 for block in components)
                            subtraces.append(float(trace))
                    if not subtraces:
                        raise AssertionError("Every actual tile must contain samples")
                    counts[iy, ix] = len(subtraces)
                    grids["sub_q25"][iy, ix] = np.quantile(subtraces, .25, method="linear")
                    grids["sub_median"][iy, ix] = np.median(subtraces)
            self._subtiles[scale] = {
                strategy: dict(trace=core._interpolate_grid(grid, centers_y,
                                                            centers_x, self.shape).astype(np.float64),
                               trace_grid=grid, subtile_counts=counts)
                for strategy, grid in grids.items()}
        return self._subtiles[scale]

    def get_bank(self, scale, lam=0, strategy=None, alpha=None):
        strategy = self.strategy if strategy is None else strategy
        alpha = self.alpha if alpha is None else float(alpha)
        self._validate(strategy, alpha)
        scale, lam = float(scale), float(lam)
        original = super().get_bank(scale, lam, mode="original")
        if alpha == 0:
            # Returning the original arrays avoids changes from public float32
            # N2 being converted back to float64 in the confidence computation.
            return original
        if strategy == "mad32" and alpha == 1:
            return super().get_bank(scale, lam, mode="robust")
        key = (scale, lam, strategy, alpha)
        if key not in self._refined_banks:
            empirical = (super()._empirical_noise(scale) if strategy == "mad32"
                         else self._subtile_fields(scale)[strategy])
            noise2 = np.maximum(original["N2"].astype(np.float64), alpha*empirical["trace"])
            confidence = self._gradient(scale)["magnitude64"] / np.sqrt(noise2+self.EPSILON)
            confidence = np.minimum(confidence.astype(np.float32), original["C"])
            self._refined_banks[key] = dict(M=original["M"], theta=original["theta"],
                                            keep=original["keep"], C=confidence,
                                            N2=noise2.astype(np.float32))
        return self._refined_banks[key]

    def diagnostics(self, scale, lam=0, strategy=None, alpha=None):
        strategy = self.strategy if strategy is None else strategy
        alpha = self.alpha if alpha is None else float(alpha)
        arrays = self.get_bank(scale, lam, strategy, alpha)
        original = super().get_bank(scale, lam, mode="original")
        empirical = (super()._empirical_noise(scale) if strategy == "mad32"
                     else self._subtile_fields(scale)[strategy])
        border = max(1, int(np.ceil(3*float(scale))))
        if min(self.shape) > 2*border:
            selection = np.s_[border:-border, border:-border]
        else:
            selection = np.s_[:,:]
        base = original["N2"][selection].astype(np.float64)
        trace = empirical["trace"][selection]
        refined = arrays["N2"][selection].astype(np.float64)
        return dict(scale=float(scale), lam=float(lam), noise_strategy=strategy,
                    alpha=alpha, tile=self.TILE, nominal_subtile=16,
                    original_noise2_median=float(np.median(base)),
                    empirical_trace_median=float(np.median(trace)),
                    refined_noise2_median=float(np.median(refined)),
                    floor_active_fraction=float(np.mean(alpha*trace>base)),
                    floor_ratio_median=float(np.median(refined/np.maximum(base,self.EPSILON))),
                    formula="N2=max(N2_original,alpha*subtile_gradient_MAD_trace)",
                    note="Lower sub-tile selection/alpha can underestimate noise; texture can inflate estimates; not an unbiased covariance model or confidence interval.")


def _self_check():
    import hashlib
    import json
    import time
    from pathlib import Path
    from optimization_bank import OptimizationBank

    started = time.perf_counter()
    rng = np.random.default_rng(7_110_102)
    image = (.15+rng.normal(0,.02,(67,79))).astype(np.float32)
    image[:,39:] += .07
    before = image.copy()
    refined = RefinedBank(image)
    old = RobustBank(image)
    original = OptimizationBank(image)
    checks = {}
    for scale in (2.,2.8):
        old_arrays = old.get_bank(scale)
        raw_arrays = original.get_bank(scale)
        new_arrays = refined.get_bank(scale, strategy="mad32", alpha=1)
        assert all(np.array_equal(old_arrays[k],new_arrays[k]) for k in old_arrays)
        for strategy in refined.STRATEGIES:
            zero = refined.get_bank(scale,strategy=strategy,alpha=0)
            assert all(np.array_equal(raw_arrays[k],zero[k]) for k in raw_arrays)
            for alpha in (.25,.5,1.):
                arrays = refined.get_bank(scale,strategy=strategy,alpha=alpha)
                assert arrays["C"].shape==image.shape
                assert all(np.isfinite(arrays[k]).all() for k in ("M","theta","C","N2"))
                assert np.all(arrays["C"]>=0) and np.all(arrays["C"]<=raw_arrays["C"])
                assert np.all(arrays["N2"]>=raw_arrays["N2"])
                assert all(np.array_equal(arrays[k],raw_arrays[k]) for k in ("M","theta","keep"))
        fields = refined._subtile_fields(scale)
        assert np.all(fields["sub_q25"]["trace"]<=fields["sub_median"]["trace"]+1e-14)
        assert all(np.isfinite(np.asarray(v)).all() for key,v in refined.diagnostics(scale,strategy="sub_q25",alpha=.5).items() if isinstance(v,(int,float)))
    checks.update(mad32_alpha1_exact_previous_bank=True, alpha0_exact_original_bank=True,
                  finite_nonnegative_confidence=True, confidence_never_above_original=True,
                  noise_floor_never_below_original=True, gradient_direction_nms_unchanged=True,
                  lower_quartile_not_above_subtile_median=True, diagnostics_finite=True,
                  truncated_tiles_supported=True)
    tiny = RefinedBank(np.arange(9,dtype=np.float32).reshape(3,3)/100)
    assert np.isfinite(tiny.get_bank(2.,strategy="sub_q25")["C"]).all()
    constant = RefinedBank(np.full((33,35),.2,np.float32))
    for strategy in constant.STRATEGIES:
        assert not constant.get_bank(2.8,strategy=strategy)["keep"].any()
    checks["tiny_and_single_tile_supported"] = True
    checks["constant_image_no_nms_candidates"] = True
    assert np.array_equal(image,before)
    checks["input_unmodified"] = True
    checks["subtile_statistics_cached"] = refined._subtile_fields(2.) is refined._subtile_fields(2.)
    for bad_strategy,bad_alpha in (("invalid",1),("mad32",-1),("mad32",float('nan'))):
        try:RefinedBank(image,bad_strategy,bad_alpha)
        except ValueError:pass
        else:raise AssertionError("Invalid strategy/alpha accepted")
    checks["invalid_configuration_rejected"] = True

    # Mechanism checks only: these images are generated here, never an
    # experimental split and never used to fit a decision threshold.
    noise = rng.normal(0,.02,(192,192)).astype(np.float32)
    correlated = ndi.gaussian_filter(rng.normal(size=noise.shape),1.25)
    correlated = (correlated*(.02/correlated.std())).astype(np.float32)
    yy,xx = np.indices(noise.shape)
    structure = (.12*(xx>=96)+.06*(yy>=96)).astype(np.float32)
    observations=[]
    for kind,observed in (("white",noise),("correlated",correlated),
                          ("white_plus_cross_steps",noise+structure)):
        bank = RefinedBank(observed)
        for scale in (2.,2.8):
            hx,hy=core.derivative_kernels(scale)
            actual = float(sum(np.var(ndi.convolve((noise if kind=='white_plus_cross_steps' else observed).astype(np.float64),k,mode='reflect')[12:-12,12:-12]) for k in (hx,hy)))
            for strategy in bank.STRATEGIES:
                arrays=bank.get_bank(scale,strategy=strategy,alpha=1.)
                observations.append(dict(case=kind,scale=scale,strategy=strategy,
                    empirical_median_over_actual_noise_trace=float(np.median(arrays['N2'][12:-12,12:-12])/actual),
                    diagnostics=bank.diagnostics(scale,strategy=strategy,alpha=1.)))
    checks["white_correlated_and_step_mechanism_observations_recorded"] = len(observations)==18
    assert all(checks.values())
    source=Path(__file__).resolve()
    report=dict(passed=True,checks=checks,mechanism_observations=observations,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                dataset_or_labels_accessed=False,elapsed_seconds=time.perf_counter()-started,
                note="Mechanism observations are descriptive. Low quantile selection is biased and alpha<1 relaxes noise rejection; no unbiasedness or statistical guarantee is claimed.")
    out=source.parents[1]/'audit'/'refined_verification.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'RefinedBank self-check passed: {len(checks)} checks; {out}',flush=True)


if __name__=='__main__':_self_check()
