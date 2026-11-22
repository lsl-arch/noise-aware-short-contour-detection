import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_NAMES = {'B0': 'Canny', 'B1': 'V / detection location',
                'B2': 'V / raised thresholds', 'B3': 'V / uniform multiplier',
                'M1': 'V+lambda D / detection', 'M2': 'V / fine location',
                'M12': 'V+lambda D / fine location'}


def read_gray(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('L')
        return np.asarray(im, dtype=np.float32)/255


def save_gray(path, a):
    Image.fromarray(np.round(np.clip(a, 0, 1)*255).astype(np.uint8)).save(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def save_case(folder, image, methods, diagnostics, scales, clean=None, truth=None, sigma=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    save_gray(folder/'input.png', image)
    arrays = {'input': image, 'V': diagnostics['V'], 'D': diagnostics['D'],
              'Veff': diagnostics['Veff'], 'residual': diagnostics['residual'],
              'M': diagnostics['base']['M'], 'theta': diagnostics['base']['theta'],
              'C_base': diagnostics['base']['C'], 'C_corrected': diagnostics['corrected']['C'],
              'N2_corrected': diagnostics['corrected']['N2']}
    if clean is not None:
        arrays['clean'] = clean; save_gray(folder/'clean.png', clean)
    if truth is not None:
        arrays['truth'] = truth; save_gray(folder/'truth.png', truth)
    if sigma is not None:
        arrays['true_sigma'] = sigma
    for name, result in methods.items():
        save_gray(folder/f'{name}.png', result['edge'])
        for key, a in result.items(): arrays[f'{name}_{key}'] = a
    np.savez_compressed(folder/'arrays.npz', **arrays)
    items = [('Input', image, 'gray')]
    if truth is not None: items.append(('Reference boundary', truth, 'gray'))
    items += [(f'{name}: {METHOD_NAMES.get(name, "fixed scale")}', r['edge'], 'gray')
              for name, r in methods.items()]
    cols = 4; rows = int(np.ceil(len(items)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5*rows), squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for ax, (title, data, cmap) in zip(axes.flat, items):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(folder/'comparison.png', dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    maps = [('Estimated sigma', np.sqrt(diagnostics['V'])),
            ('Dispersion D', diagnostics['D']), ('Effective sigma', np.sqrt(diagnostics['Veff'])),
            ('Detection scale (M12)', np.ma.masked_equal(methods['M12']['sd'], 0)),
            ('Location scale (M12)', np.ma.masked_equal(methods['M12']['sl'], 0)),
            ('Max confidence (M12)', diagnostics['corrected']['C'].max(axis=0))]
    for ax, (title, data) in zip(axes.flat, maps):
        limits = dict(vmin=min(scales), vmax=max(scales)) if 'scale (M12)' in title else {}
        obj = ax.imshow(data, cmap='viridis', interpolation='nearest', **limits)
        ax.set_title(title); ax.axis('off')
        fig.colorbar(obj, ax=ax, shrink=.75)
    fig.tight_layout(); fig.savefig(folder/'diagnostics.png', dpi=130); plt.close(fig)
