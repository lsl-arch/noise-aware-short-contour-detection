"""第二课方法：局部噪声估计、尺度空间、幅值NMS和互近邻候选链。"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass
class Config:
    scales: tuple = (0.7, 1.4, 2.8)
    tile: int = 32
    lam: float = 1.0
    low: float = 3.0
    high: float = 5.0
    angle_deg: float = 20.0
    threshold_factor: float = 1.2
    variance_factor: float = 1.5
    canny_sigma: float = 1.4
    canny_low: int = 30
    canny_high: int = 60
    epsilon: float = 1e-12

    def validate(self):
        if not self.scales or any(s <= 0 for s in self.scales):
            raise ValueError('scales must be positive')
        if list(self.scales) != sorted(set(self.scales)):
            raise ValueError('scales must be strictly increasing')
        if self.tile < 4 or self.tile % 2:
            raise ValueError('tile must be even and >= 4')
        if not 0 < self.low <= self.high:
            raise ValueError('require 0 < low <= high')
        if self.lam < 0 or self.epsilon <= 0 or not 0 <= self.angle_deg <= 90:
            raise ValueError('invalid lambda, epsilon or angle')
        if self.threshold_factor < 1 or self.variance_factor < 1:
            raise ValueError('control factors must be >= 1')
        if self.canny_sigma <= 0 or not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError('invalid Canny configuration')

    @property
    def border(self):
        return int(math.ceil(3 * max(self.scales)))

    def to_dict(self):
        return asdict(self)


def mad(a):
    return np.median(np.abs(a - np.median(a)))


def _interpolate_grid(grid, ys, xs, shape):
    # 使用实际区域中心做双线性插值，边缘采用最近中心值。
    xx, yy = np.arange(shape[1]), np.arange(shape[0])
    horizontal = np.array([np.interp(xx, xs, row) for row in grid])
    return np.array([np.interp(yy, ys, horizontal[:, j])
                     for j in range(shape[1])], dtype=np.float32).T


def estimate_noise(image, tile=32):
    """返回V、D和高频残差。末尾不足32的区域只使用有效像素。"""
    image = np.asarray(image, dtype=np.float32)
    kernel = np.array([[1, -1], [-1, 1]], dtype=np.float32) / 2
    residual = ndi.convolve(image, kernel, mode='reflect')
    h, w = image.shape
    y_starts, x_starts = list(range(0, h, tile)), list(range(0, w, tile))
    v_grid = np.zeros((len(y_starts), len(x_starts)), dtype=np.float32)
    d_grid = np.zeros_like(v_grid)
    ys, xs = [], []
    for y in y_starts:
        ys.append((y + min(y + tile, h) - 1) / 2)
    for x in x_starts:
        xs.append((x + min(x + tile, w) - 1) / 2)
    for iy, y in enumerate(y_starts):
        for ix, x in enumerate(x_starts):
            block = residual[y:min(y + tile, h), x:min(x + tile, w)]
            # 完整区域是四个16×16子块。截断区域将实际像素近似均分。
            subvars = []
            for rows in np.array_split(block, 2, axis=0):
                for sub in np.array_split(rows, 2, axis=1):
                    if sub.size:
                        subvars.append(float(mad(sub) / 0.67449) ** 2)
            v = np.median(subvars)
            v_grid[iy, ix] = v
            d_grid[iy, ix] = 1.4826 * np.median(np.abs(np.array(subvars) - v))
    return (_interpolate_grid(v_grid, ys, xs, image.shape),
            _interpolate_grid(d_grid, ys, xs, image.shape), residual)


def derivative_kernels(scale):
    """半径ceil(3s)，离散高斯归一化，导数核零均值校正。"""
    r = int(math.ceil(3 * scale))
    q = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-q*q / (2 * scale*scale))
    g /= g.sum()
    dg = -q * g / (scale*scale)
    hx = np.outer(g, dg)
    hy = hx.T.copy()
    hx -= hx.mean()
    hy -= hy.mean()
    return hx, hy


def build_scale_bank(image, variance, scales, epsilon=1e-12):
    """M、theta、C、N2均为[S,H,W]。N2传播使用同一最终核的平方。"""
    M, theta, C, N2 = [], [], [], []
    for s in scales:
        hx, hy = derivative_kernels(s)
        gx = ndi.convolve(image.astype(np.float64), hx, mode='reflect')
        gy = ndi.convolve(image.astype(np.float64), hy, mode='reflect')
        magnitude = np.hypot(gx, gy)
        noise2 = ndi.convolve(variance.astype(np.float64), hx*hx + hy*hy,
                              mode='reflect')
        M.append(magnitude)
        theta.append(np.arctan2(gy, gx))
        N2.append(noise2)
        C.append(magnitude / np.sqrt(np.maximum(noise2, 0) + epsilon))
    return {k: np.asarray(v, dtype=np.float32)
            for k, v in [('M', M), ('theta', theta), ('C', C), ('N2', N2)]}


def nms(magnitude, theta):
    """四方向幅值NMS；一侧严格比较消除平顶重复。最外一圈不输出。"""
    angles = np.mod(np.degrees(theta), 180)
    bins = (np.floor((angles + 22.5) / 45).astype(int)) % 4
    padded = np.pad(magnitude, 1, mode='edge')
    h, w = magnitude.shape
    keep = np.zeros_like(magnitude, dtype=bool)
    for k, (dy, dx) in enumerate([(0, 1), (1, 1), (1, 0), (1, -1)]):
        forward = padded[1+dy:1+dy+h, 1+dx:1+dx+w]
        backward = padded[1-dy:1-dy+h, 1-dx:1-dx+w]
        keep |= ((bins == k) & (magnitude >= forward) &
                 (magnitude > backward) & (magnitude > 0))
    keep[[0, -1], :] = False
    keep[:, [0, -1]] = False
    return keep


def make_candidates(bank, low):
    result = []
    for si in range(len(bank['M'])):
        keep = nms(bank['M'][si], bank['theta'][si]) & (bank['C'][si] >= low)
        yx = np.column_stack(np.nonzero(keep))
        result.append({'yx': yx, 'score': bank['C'][si][keep],
                       'theta': bank['theta'][si][keep]})
    return result


def angular_difference(a, b):
    return np.abs((a - b + np.pi/2) % np.pi - np.pi/2)


def mutual_matches(a, b, radius, angle_deg):
    """在距离和轴向角度合格的备选中取互近邻，距离并列按像素顺序。"""
    if len(a['yx']) == 0 or len(b['yx']) == 0:
        return []
    nearby = cKDTree(b['yx']).query_ball_point(a['yx'], radius)
    best_a = np.full(len(a['yx']), -1, dtype=int)
    best_b = np.full(len(b['yx']), -1, dtype=int)
    dist_b = np.full(len(b['yx']), np.inf)
    for i, js in enumerate(nearby):
        options = []
        for j in js:
            if angular_difference(a['theta'][i], b['theta'][j]) <= np.radians(angle_deg):
                dist = float(np.sum((a['yx'][i] - b['yx'][j])**2))
                options.append((dist, j))
                if dist < dist_b[j] or (dist == dist_b[j] and i < best_b[j]):
                    dist_b[j], best_b[j] = dist, i
        if options:
            best_a[i] = min(options)[1]
    return [(i, int(j)) for i, j in enumerate(best_a)
            if j >= 0 and best_b[j] == i]


def build_chains(candidates, scales, angle_deg=20):
    nodes = [(s, j) for s, c in enumerate(candidates) for j in range(len(c['yx']))]
    successor, predecessor = {}, set()
    for s in range(len(scales)-1):
        for i, j in mutual_matches(candidates[s], candidates[s+1],
                                   math.ceil(scales[s+1]), angle_deg):
            successor[(s, i)] = (s+1, j)
            predecessor.add((s+1, j))
    chains = []
    for node in nodes:
        if node in predecessor:
            continue
        chain = [node]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        chains.append(chain)
    return chains


def select_chains(candidates, chains, scales, high, shape, fine_location):
    edge = np.zeros(shape, bool)
    sd, sl = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    score = np.zeros(shape, np.float32)
    for chain in chains:
        strong = [(s, j) for s, j in chain if candidates[s]['score'][j] >= high]
        if not strong:
            continue
        ds, dj = min(strong)
        ls, lj = min(chain) if fine_location else (ds, dj)
        y, x = candidates[ls]['yx'][lj]
        # 完全同像素去重：优先最细定位尺度，其次最细检测尺度。
        if edge[y, x] and (sl[y, x], sd[y, x]) <= (scales[ls], scales[ds]):
            continue
        edge[y, x] = True
        sd[y, x], sl[y, x] = scales[ds], scales[ls]
        score[y, x] = candidates[ls]['score'][lj]
    return {'edge': edge, 'sd': sd, 'sl': sl, 'score': score}


def link_and_select(candidates, scales, high, shape, angle_deg=20, fine_location=True):
    chains = build_chains(candidates, scales, angle_deg)
    return select_chains(candidates, chains, scales, high, shape, fine_location)


def run_methods(image, cfg):
    """返回7个对照、3个固定尺度基线、中间图及逐方法运行时间。"""
    import cv2
    import time
    cfg.validate()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError('input must be a finite 2D grayscale array')
    if min(image.shape) <= 2*cfg.border+1:
        raise ValueError('image is too small for the largest scale')
    t = time.perf_counter()
    V, D, residual = estimate_noise(image, cfg.tile)
    estimation_ms = (time.perf_counter()-t)*1000
    methods, diagnostics, timings = {}, {}, {}
    # 相同方差的两个定位版本共享bank、候选和链，隔离位置选择影响。
    for tag, variance, factor, variants in [
        ('base', V, 1, [('B1', False), ('M2', True)]),
        ('corrected', V+cfg.lam*D, 1, [('M1', False), ('M12', True)]),
        ('threshold', V, cfg.threshold_factor, [('B2', False)]),
        ('multiplier', V*cfg.variance_factor, 1, [('B3', False)])]:
        t = time.perf_counter()
        bank = build_scale_bank(image, variance, cfg.scales, cfg.epsilon)
        candidates = make_candidates(bank, cfg.low*factor)
        chains = build_chains(candidates, cfg.scales, cfg.angle_deg)
        common_ms = (time.perf_counter()-t)*1000 + estimation_ms
        for name, fine in variants:
            t = time.perf_counter()
            methods[name] = select_chains(candidates, chains, cfg.scales,
                                          cfg.high*factor, image.shape, fine)
            timings[name] = common_ms + (time.perf_counter()-t)*1000
        if tag in ('base', 'corrected'):
            diagnostics[tag] = bank
        if tag == 'base':
            for si, scale in enumerate(cfg.scales):
                edge = nms(bank['M'][si], bank['theta'][si]) & (bank['C'][si] >= cfg.high)
                methods[f'F_s{scale:g}'] = {'edge': edge}
    t = time.perf_counter()
    # OpenCV Canny需要uint8；明确保存此量化基线的定义。
    u8 = np.round(np.clip(image, 0, 1)*255).astype(np.uint8)
    r = math.ceil(3*cfg.canny_sigma)
    smooth = cv2.GaussianBlur(u8, (2*r+1, 2*r+1), cfg.canny_sigma,
                              borderType=cv2.BORDER_REFLECT)
    edge = cv2.Canny(smooth, cfg.canny_low, cfg.canny_high, apertureSize=3, L2gradient=True)>0
    methods['B0'] = {'edge': edge}
    order = ['B0', 'B1', 'B2', 'B3', 'M1', 'M2', 'M12'] + [f'F_s{s:g}' for s in cfg.scales]
    methods = {name: methods[name] for name in order}
    timings['B0'] = (time.perf_counter()-t)*1000
    diagnostics.update(V=V, D=D, Veff=V+cfg.lam*D, residual=residual)
    return methods, diagnostics, timings
