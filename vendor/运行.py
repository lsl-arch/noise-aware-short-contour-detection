"""Fixed detector entry point. Only the observed image enters the detector."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts'))
from refined_bank import RefinedBank
from spatial_methods import connected_edge
METHOD = json.loads((ROOT/'method.json').read_text(encoding='utf-8'))

def detect(image):
    """Image -> binary edges; no clean image, GT, or noise-family input."""
    bank = RefinedBank(np.asarray(image, dtype=np.float32))
    b = bank.get_bank(METHOD['scale'], strategy=METHOD['strategy'], alpha=METHOD['alpha'])
    return connected_edge(b['keep'], b['C'], METHOD['high'], min_size=METHOD['min_size'], hysteresis=False)

def read_photo(path):
    with Image.open(path) as src:
        rgb = np.asarray(ImageOps.exif_transpose(src).convert('RGB'), dtype=np.float32)/255.
    if min(rgb.shape[:2]) < 32:
        raise ValueError('图片宽高都必须至少为32像素。')
    gray = rgb @ np.array([.299, .587, .114], dtype=np.float32)
    return rgb, gray

def main():
    parser = argparse.ArgumentParser(description='固定参数边缘检测；支持单图或文件夹（不递归）。')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    extensions = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
    if not args.input.exists():
        parser.error('输入路径不存在。')
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in extensions) if args.input.is_dir() else [args.input]
    if not paths:
        parser.error('未找到支持的图片。')
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        start = time.perf_counter()
        rgb, gray = read_photo(path)
        pred = detect(gray)
        destination = args.output/(path.stem+'_'+hashlib.sha256(path.name.encode('utf-8')).hexdigest()[:8])
        destination.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(np.clip(gray,0,1)*255)).save(destination/'input_gray.png')
        Image.fromarray(np.uint8(pred)*255).save(destination/'edges.png')
        overlay = np.uint8(np.clip(rgb,0,1)*255)
        overlay[pred] = [255,0,0]
        Image.fromarray(overlay).save(destination/'overlay.png')
        elapsed = time.perf_counter()-start
        info = dict(input_name=path.name, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    input_shape=list(gray.shape), image_preprocessing='EXIF transpose; RGB [0,1]; gray=.299R+.587G+.114B; no extra synthetic noise',
                    method=METHOD, edge_pixels=int(pred.sum()), detector_input='observed grayscale image only',
                    elapsed_seconds=elapsed, output_files=['input_gray.png','edges.png','overlay.png'])
        (destination/'info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{path.name}: {int(pred.sum())} edge pixels -> {destination}', flush=True)

if __name__ == '__main__':
    main()
