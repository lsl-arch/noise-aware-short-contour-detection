# Noise-Aware Component Selection for Short-Contour Detection in Low-Light Images

This repository contains the frozen implementation, experiment protocol, and aggregate evidence for a training-free method that selects edge components using an empirical noise floor, local axial-orientation coherence, and cross-scale support. It is intended to support inspection and reproduction of the experiments reported in the accompanying manuscript.

## Scope and Reproducibility

The final run was executed on Windows 11 with Python 3.12.0, an Intel Core i9-13900HX CPU, NumPy 2.5.4, SciPy 1.18.1, Pillow 12.3.0, OpenCV-Contrib 4.14.0.94, and PyTorch 2.15.0+cpu. OpenCV was limited to one thread and PyTorch to four. The frozen run completed in about 26 minutes on that system. The complete package versions and verification record are provided in [`results/final_20261119/`](results/final_20261119/).

The repository intentionally excludes BSDS500 images and annotations, the Structured Forests model, and the DexiNed checkpoint. These assets are distributed by their respective sources and may have separate terms. The runner validates source hashes against [`frozen_protocol.json`](frozen_protocol.json); use `verify_models.ps1` to check downloaded model hashes before running all baselines. Do not commit downloaded datasets or weights to this repository.

## Method

For an input grayscale image, the detector computes Gaussian-derivative responses at scales $s=2.0$ and $s=2.8$. A tiled median absolute deviation (MAD) estimate supplies an empirical local noise-energy floor. The normalized score is

$$
C_s = \frac{M_s}{\sqrt{\max\{N_{s,0}^2,\,\alpha\widetilde E_s\}+10^{-12}}},\qquad \alpha=0.5.
$$

The implementation uses zero-mean derivative kernels with radius $\lceil 3s\rceil$, reflection padding, 32-by-32 MAD tiles, bilinear interpolation between tile centers, four-direction nonmaximum suppression, and an initial normalized-response threshold of 2.5. Eight-connected components are then screened. Local axial coherence averages doubled-angle orientation vectors in a 3-by-3 neighborhood; cross-scale support requires a nearby response at the other scale whose axial difference is at most 30 degrees.

The frozen full selector rejects components smaller than 4 pixels. A component of 4--31 pixels is retained only when its median normalized response is at least 3.75, its mean local coherence is at least 0.85, and its mean cross-scale support is at least 0.50. Components of at least 32 pixels require mean cross-scale support of at least 0.50. The length baseline uses the same primary-scale candidate map and retains components of at least 32 pixels. Exact variants and settings are in the protocol JSON.

## Data and Operating Points

The positive synthetic set has 384 images at 128 by 128 pixels. It crosses four shapes (bar, circle, L-shape, and blurred semicircle), four contrast amplitudes (0.01, 0.02, 0.04, 0.08), and six artifact families (constant-variance Gaussian noise, spatially varying Gaussian noise, correlated noise, banding, quantization, and JPEG degradation), with four realizations per combination and nominal noise scales balanced over 0.01, 0.02, 0.03, and 0.04. A separate set of 384 edge-free controls uses the same artifact families on a sloped background. The final synthetic seeds differ from development seeds, but the shape and degradation families are shared; this is a fresh-seed evaluation, not an out-of-distribution test.

Natural-image evaluation uses all 200 BSDS500 test scenes and their original human boundary maps. For grayscale intensity $Y\in[0,1]$, the simulated low-light input is

$$
J = aY^{1.5}, \qquad I = J + \sqrt{0.004^2 + 0.0006J}\,Z.
$$

where $Z$ is standard normal noise generated deterministically from the scene identifier. The conditions use $a=0.25$ (Low) and $a=0.10$ (Darker), with the same noise field for both conditions. These scenes were examined earlier in the project, so this is a fixed re-evaluation of a previously viewed benchmark, not an unseen-scene test. It is also simulated low light, not a collection of real low-light camera captures.

The final configuration was selected on previously used synthetic/control development cases and 20 BSDS validation scenes under Low illumination. Eligible candidates had to achieve natural-image F1 no lower than the length baseline and produce no more than 20 nonempty outputs among 192 development controls; among eligible candidates, selection maximized synthetic F1 plus 0.2 times short-edge recall. Thresholds and the selected configuration were frozen before scoring the final 384 synthetic images, 384 controls, and the reported BSDS test re-evaluation. No final-set result was used to tune a parameter.

### Baseline preprocessing and thresholds

| Method | Preprocessing and fixed threshold |
| --- | --- |
| Canny | Clip grayscale to [0, 1], convert to 8-bit, Gaussian blur with sigma 1, L2-gradient Canny; high threshold 40 and low threshold 0.4 times high. |
| CLAHE + Canny | CLAHE with an 8-by-8 tile grid and clip limit 2, then the same blur and Canny rule; high threshold 120 and low threshold 0.4 times high. |
| Structured Forests | Grayscale replicated to three float channels; OpenCV `ximgproc`; response thinning with radius 2; threshold 0.08. |
| DexiNed | Grayscale replicated to three channels, resized to 512-by-512, channel mean subtraction, author-released extended checkpoint, sigmoid on fused output, per-image min-max scaling, resize to input dimensions, OpenCV thinning radius 2; threshold 0.01. |

The four baseline thresholds were selected from a finite grid on 20 Low-condition validation scenes and reused unchanged in Darker. Structured Forests uses the OpenCV model distributed in `opencv_extra`; its exact training split was not independently verified. DexiNed uses the author-released extended implementation and BIPED-trained checkpoint. Checkpoint and source references are listed under [External assets](#external-assets).

## Evaluation Protocol

- Exclude a 13-pixel border from scoring.
- Match predicted and reference edge pixels one-to-one within a 2-pixel Euclidean tolerance. The assignment first maximizes match cardinality, then minimizes total matched distance.
- For synthetic data, compute precision, recall, and F1 from pooled pixel counts (micro aggregation).
- Define short ground-truth edges before cropping as eight-connected annotated components containing at most 32 pixels; report pooled recall over their pixels.
- For edge-free controls, report how many of the 384 images contain at least one predicted edge in the evaluation window.
- For BSDS500, score each human annotation separately, average annotations within each image, then average images (macro aggregation). The fixed-threshold F1 reported here is not official BSDS ODS, OIS, or AP.
- Paired uncertainty intervals use 3,000 image-level bootstrap resamples. Synthetic pooled metrics are recomputed within each resample.

## Reported Results

| Evaluation | Length baseline | Full method | Difference |
| --- | ---: | ---: | ---: |
| Synthetic micro F1, 384 images | 0.37307 | 0.38740 | +0.01433 |
| Short-edge recall, 384 images | 25.54% | 35.34% | +9.80 percentage points |
| Edge-free images with any output, 384 controls | 29 (7.55%) | 38 (9.90%) | +9 images |
| BSDS500 simulated Low mean F1, 200 scenes | 0.37104 | 0.37344 | +0.00240 (+0.24 points) |
| BSDS500 simulated Darker mean F1, 200 scenes | 0.37110 | 0.37437 | +0.00327 (+0.33 points) |

These results show a trade-off: the method recovers more short boundaries while producing detections on more edge-free controls. Its natural-image gains over the length baseline are small. The ablation results also do not support claiming that every module improves every metric; inspect `summary.csv` and `paired_uncertainty.json` before making a broader claim.

## Installation

Use 64-bit Python 3.12. In PowerShell from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Select `.venv\Scripts\python.exe` as the VS Code interpreter. On a machine that cannot resolve the pinned PyTorch version from its configured package index, install a compatible CPU build for Python 3.12 from the official PyTorch installation selector, then install the remaining packages in `requirements.txt`. The recorded reference run used CPU inference; CUDA is not required.

## Datasets and Model Files

1. Download BSDS500 from the official Berkeley Segmentation Dataset and Benchmark resource page: <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html>.
2. Extract it so that the repository contains `data/BSDS500/images/test/*.jpg` and `data/BSDS500/groundTruth/test/*.mat`. The runner requires exactly 200 test JPG files.
3. Download the OpenCV Structured Forests model from the OpenCV Extra repository: <https://github.com/opencv/opencv_extra/blob/4.x/testdata/cv/ximgproc/model.yml.gz>. Store it as `models/structured_forest.yml.gz`. For Windows paths containing non-ASCII characters, also save an identical copy as `models_ascii/sf.yml.gz`; verify it against SHA-256 `fe3586bca5ec124b73fa769c2b54e195e4526d6ceeddf161fad96bf68a65bf17`.
4. Download the extended DexiNed checkpoint from the author project reference at <https://github.com/xavysp/DexiNed> (checkpoint link is in the project README) and store it as `models/dexined.pth`. Expected SHA-256: `bd4c603ef71113b447bffb72a73b9a54ca890e78f4bc8f34cb73f1efb2681e74`.
5. The model architecture used by this run is supplied in `models/dexined_model.py`; expected SHA-256: `d417d0db386205ede64267ae438f9bcb38f84f3832bced1d41bd2e011c5fab53`.

The weights are not committed. After downloading them, run `powershell -ExecutionPolicy Bypass -File .\verify_models.ps1` from the repository root. The archived `protocol.json` also records expected model hashes and provenance.

## Running the Experiment

The complete run writes per-image records, summary metrics, and compressed predictions to the selected output directory. It takes substantially longer and uses more disk space than the unit-sized example because it evaluates 384 synthetic images, 384 controls, two conditions for 200 BSDS scenes, seven internal configurations, and four external baselines.

```powershell
$env:PYTHONUTF8='1'
$env:LESSON5_SKIP_DEXINED='0'
& .\.venv\Scripts\python.exe -X utf8 -u .\run_final.py `
  --bsds-root .\data\BSDS500 `
  --output .\results\reproduction_local
```

The run stops if the output directory already contains `summary.json`; choose a new `--output` path for each run. To omit the DexiNed baseline when its checkpoint is unavailable, set `$env:LESSON5_SKIP_DEXINED='1'`. This produces an incomplete baseline comparison and must not be presented as the full reported experiment.

## Repository Contents

| Path | Contents |
| --- | --- |
| `run_final.py` | Frozen full experiment entry point. |
| `method.py`, `baselines.py` | Proposed detector, ablations, and external baseline preprocessing. |
| `frozen_protocol.json` | Frozen thresholds, development rule, dataset roles, source hashes, and evaluation details. |
| `vendor/` | Local modules required by the frozen runner and synthetic-data generator. |
| `models/` | DexiNed architecture; downloaded weights are deliberately excluded. |
| `results/final_20261119/summary.csv` | Compact metrics for all stages and methods. |
| `results/final_20261119/summary.json` | Aggregate metrics and method-level counts. |
| `results/final_20261119/per_image.json` | Per-image measurements used to recompute aggregate results. |
| `results/final_20261119/paired_uncertainty.json` | Paired bootstrap intervals for key comparisons. |
| `results/final_20261119/environment.json` | Sanitized runtime and package information. |
| `results/final_20261119/protocol.json` | Exact protocol snapshot copied from the final run. |
| `results/final_20261119/reproduction_verification.json` | Verification summary for 9,776 records and predictions. |

Generated full-run predictions, downloaded datasets, checkpoints, virtual environments, and caches are excluded by `.gitignore`.

## Verification of the Archived Run

The archived final run contains 9,776 per-image/per-method records. Its recorded verification reports exact agreement for the 768 synthetic and edge-free input arrays, agreement of the natural degraded inputs within $10^{-7}$ across the checked platform variation, exact equality for all ground-truth arrays and all seven internal-method masks, and successful recomputation of all aggregate summaries. DexiNed ran on all 400 natural-image inputs. See `reproduction_verification.json` for the machine-readable evidence. These checks document the archived run; running on a different operating system or numerical-library build can yield small floating-point differences near thresholds.

## External Assets

- BSDS500: <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html>
- OpenCV Extra Structured Forests model: <https://github.com/opencv/opencv_extra/blob/4.x/testdata/cv/ximgproc/model.yml.gz>
- DexiNed authors' code and model instructions: <https://github.com/xavysp/DexiNed>
- DexiNed author checkpoint reference: <https://drive.google.com/file/d/1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu/view>

Review the source project, dataset, and checkpoint terms before redistribution. This repository does not grant rights to third-party data or weights.

## Limitations

The method is a fixed, training-free component selector rather than a learned low-light enhancement or edge-detection network. Synthetic noise families do not capture all camera pipelines. BSDS500 inputs are synthetically darkened, and the project history includes earlier inspection of the benchmark scenes. The custom fixed-threshold F1 is useful for paired analysis within this experiment but should not be compared as if it were an official BSDS benchmark score. The false-output increase on empty controls and the small natural-image gains should be reported alongside short-edge recall.
