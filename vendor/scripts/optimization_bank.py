"""Cached parameter variants of the classroom noise-aware edge detector.

This module never opens a dataset or annotation. Gradients and propagated
noise energy use the original final discrete derivative kernels, including
their zero-mean correction. M/theta/C/N2 retain the original float32 outputs;
confidence is computed from the float64 intermediate magnitude and energy.
"""
import numpy as np
from scipy import ndimage as ndi

from bsds_support import core, fast_chains


class OptimizationBank:
    SCALES = (.7, 1.4, 2.8)
    TILE = 32
    FLOOR_SIGMA = 1 / 255
    EPSILON = 1e-12
    ANGLE_DEG = 20

    def __init__(self, image):
        self.image = np.asarray(image, dtype=np.float32)
        if self.image.ndim != 2 or not np.isfinite(self.image).all():
            raise ValueError('image must be a finite 2D grayscale array')
        self.shape = self.image.shape
        self.V, self.D, _ = core.estimate_noise(self.image, tile=self.TILE)
        self._gradients = {}
        self._variances = {}
        self._banks = {}
        self._candidates = {}

    def _gradient(self, scale):
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be positive and finite')
        if scale not in self._gradients:
            hx, hy = core.derivative_kernels(scale)
            gx = ndi.convolve(self.image.astype(np.float64), hx, mode='reflect')
            gy = ndi.convolve(self.image.astype(np.float64), hy, mode='reflect')
            magnitude64 = np.hypot(gx, gy)
            theta = np.arctan2(gy, gx).astype(np.float32)
            magnitude = magnitude64.astype(np.float32)
            self._gradients[scale] = dict(
                magnitude64=magnitude64, M=magnitude, theta=theta,
                kernel2=hx*hx + hy*hy,
                keep=core.nms(magnitude, theta))
        return self._gradients[scale]

    def _variance(self, lam):
        lam = float(lam)
        if not np.isfinite(lam) or lam < 0:
            raise ValueError('lambda must be nonnegative and finite')
        if lam not in self._variances:
            self._variances[lam] = np.maximum(
                self.V + lam*self.D, self.FLOOR_SIGMA**2)
        return self._variances[lam]

    def get_bank(self, scale, lam=0):
        """Return M/theta/C/N2/keep for one scale and reliability weight."""
        scale, lam = float(scale), float(lam)
        key = (scale, lam)
        if key not in self._banks:
            gradient = self._gradient(scale)
            variance = self._variance(lam)
            noise2 = ndi.convolve(variance.astype(np.float64),
                                  gradient['kernel2'], mode='reflect')
            confidence = (gradient['magnitude64'] /
                          np.sqrt(np.maximum(noise2, 0) + self.EPSILON))
            self._banks[key] = dict(
                M=gradient['M'], theta=gradient['theta'],
                C=confidence.astype(np.float32),
                N2=noise2.astype(np.float32), keep=gradient['keep'])
        return self._banks[key]

    def fixed(self, scale, lam, high):
        """Single-scale NMS with the specified confidence threshold."""
        high = float(high)
        if not np.isfinite(high) or high <= 0:
            raise ValueError('high must be positive and finite')
        bank = self.get_bank(scale, lam)
        return bank['keep'] & (bank['C'] >= high)

    def multiscale_conf(self, lam, high, low_ratio=.6, rho=.8):
        """Original PPT three-scale chains and confidence-gated location."""
        lam, high, low_ratio, rho = map(float, (lam, high, low_ratio, rho))
        if not np.isfinite(high) or high <= 0:
            raise ValueError('high must be positive and finite')
        if not 0 < low_ratio <= 1 or not 0 <= rho <= 1:
            raise ValueError('require 0 < low_ratio <= 1 and 0 <= rho <= 1')
        key = (lam, high, low_ratio)
        low = high*low_ratio
        if key not in self._candidates:
            candidates = []
            for scale in self.SCALES:
                bank = self.get_bank(scale, lam)
                keep = bank['keep'] & (bank['C'] >= low)
                candidates.append(dict(
                    yx=np.column_stack(np.nonzero(keep)),
                    score=bank['C'][keep], theta=bank['theta'][keep]))
            self._candidates[key] = candidates, fast_chains(candidates)
        candidates, chains = self._candidates[key]
        output = np.zeros(self.shape, dtype=bool)
        location_threshold = max(low, rho*high)
        for chain in chains:
            strong = [(s, j) for s, j in chain
                      if candidates[s]['score'][j] >= high]
            if not strong:
                continue
            ds, dj = min(strong)
            choices = [(s, j) for s, j in chain
                       if s <= ds and candidates[s]['score'][j] >= location_threshold]
            ls, lj = min(choices) if choices else (ds, dj)
            y, x = candidates[ls]['yx'][lj]
            output[y, x] = True
        return output
