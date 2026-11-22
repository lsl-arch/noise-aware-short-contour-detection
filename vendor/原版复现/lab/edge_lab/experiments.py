"""可复算的合成数据，以及带一对一匹配的教学评价。"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def synthetic_scenes(size=160):
    """五种已知结构。GT为整数栅格参考边界，不从任一待比较检测器产生。"""
    if size < 64:
        raise ValueError('synthetic size must be >= 64')
    y, x = np.indices((size, size)); mid = size//2
    cases = []
    image = np.full((size, size), .15, np.float32)
    image[:, mid:] = .28
    gt = np.zeros_like(image, bool); gt[:, mid] = True
    cases.append(('step', image, gt))
    image = np.full((size, size), .15, np.float32)
    left, right = mid-2, mid+2
    image[:, left:right] = .24
    gt = np.zeros_like(image, bool); gt[:, [left, right]] = True
    cases.append(('thin_bar', image, gt))
    image = np.full((size, size), .15, np.float32)
    image[mid:, mid:] = .28
    gt = np.zeros_like(image, bool)
    gt[mid, mid:] = True; gt[mid:, mid] = True
    cases.append(('corner', image, gt))
    image = np.full((size, size), .15, np.float32)
    # 隐式圆距离的0.5像素带作为已知轮廓。
    radius = size*.24
    distance = np.hypot(x-mid, y-mid)
    image[distance <= radius] = .28
    gt = np.abs(distance-radius) <= .5
    cases.append(('circle', image, gt))
    image = np.full((size, size), .15, np.float32)
    image[:, mid-4:] += .07
    image[:, mid+4:] += .07
    gt = np.zeros_like(image, bool); gt[:, [mid-4, mid+4]] = True
    cases.append(('nearby_edges', image, gt))
    return cases


def noise_profiles(shape):
    """分区sigma为.01到.04，过渡在宽度5/12到7/12；均匀条件匹配平均方差。"""
    h, w = shape
    t = np.clip((np.arange(w)/max(w-1, 1) - 5/12)/(2/12), 0, 1)
    spatial = np.broadcast_to(.01 + .03*t, (h, w)).copy()
    uniform = np.full((h, w), np.sqrt(np.mean(spatial**2)))
    return {'uniform': uniform, 'spatial': spatial}


def degraded_pairs(clean, seed):
    field = np.random.default_rng(seed).standard_normal(clean.shape)
    for name, sigma in noise_profiles(clean.shape).items():
        # 不裁剪浮点实验输入，维持加性高斯模型；PNG仅用于显示。
        yield name, (clean + field*sigma).astype(np.float32), sigma.astype(np.float32)


def valid_region(shape, border):
    valid = np.zeros(shape, bool)
    if min(shape) > 2*border:
        valid[border:shape[0]-border, border:shape[1]-border] = True
    return valid


def match_boundaries(pred, truth, tolerance=2.0):
    """最大匹配数优先，其次最小总距离，避免多个预测重复认领同一GT。"""
    a, b = np.column_stack(np.nonzero(pred)), np.column_stack(np.nonzero(truth))
    if not len(a) or not len(b):
        return np.empty(0), len(a), len(b)
    rows, cols, costs = [], [], []
    neighbors = cKDTree(b).query_ball_point(a, tolerance)
    for i, js in enumerate(neighbors):
        for j in js:
            rows.append(i); cols.append(j)
            costs.append(float(np.linalg.norm(a[i]-b[j])) + 1e-8)
    # 每个预测独有一个虚拟GT。足够大的惩罚确保最大匹配优先。
    penalty = (min(len(a), len(b))+1)*(tolerance+1)
    rows.extend(range(len(a))); cols.extend(len(b)+np.arange(len(a)))
    costs.extend([penalty]*len(a))
    graph = coo_matrix((costs, (rows, cols)), shape=(len(a), len(b)+len(a))).tocsr()
    ia, ib = min_weight_full_bipartite_matching(graph)
    real = ib < len(b)
    distances = np.linalg.norm(a[ia[real]]-b[ib[real]], axis=1)
    return distances, len(a), len(b)


def evaluate(pred, truth, clean, border=9, tolerance=2):
    valid = valid_region(pred.shape, border)
    distances, n_pred, n_gt = match_boundaries(pred & valid, truth & valid, tolerance)
    tp = len(distances)
    precision = tp/n_pred if n_pred else 0.0
    recall = tp/n_gt if n_gt else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    # 合成图的结构外平坦区域，所有方法共享。剔除结构附近最大核半径。
    flat = valid & (ndi.distance_transform_edt(~truth) > border+1)
    false_flat = np.count_nonzero(pred & flat)/np.count_nonzero(flat) if flat.any() else None
    return dict(precision=precision, recall=recall, f1=f1,
                matched_mean_px=float(distances.mean()) if tp else None,
                matched_p95_px=float(np.percentile(distances, 95)) if tp else None,
                flat_response_rate=false_flat, matched=tp, predicted=n_pred,
                truth=n_gt, valid_pixels=int(valid.sum()))
