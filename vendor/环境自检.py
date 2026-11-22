"""Print the actual current runtime; CPU-only, no CUDA requirement."""
import importlib.metadata as m
import json, platform, sys
required={'numpy':'2.5.4','scipy':'1.18.1','Pillow':'12.3.0','opencv-python-headless':'4.14.0.94'}
packages={};missing=[]
for name,expected in required.items():
    try:
        actual=m.version(name);packages[name]={'installed':actual,'verified':expected,'matches':actual==expected}
    except m.PackageNotFoundError:missing.append(name)
result=dict(python=sys.version,python_executable=sys.executable,platform=platform.platform(),machine=platform.machine(),
            processor=platform.processor(),packages=packages,missing=missing,gpu_required=False,
            note='Verified on Python 3.12.0 Windows x64. Results are checked by reproduce.py --check; CPU/platform differences may change floating point rounding.')
print(json.dumps(result,ensure_ascii=False,indent=2))
if missing:raise SystemExit(1)
