# Auto Icon Vectorizer

Convert a cropped raster UI icon into clean inline SVG HTML.

Auto Icon Vectorizer is a small-image icon cleanup and vectorization tool. It
takes one cropped raster image that already contains a single UI icon, removes
the background, traces the recovered foreground mask, and returns HTML with an
inline SVG that can be placed directly in a web page.

The problem it solves is narrower than general image vectorization. Existing
open source tools are already strong at tracing clean bitmaps, scans, logos,
pixel art, or full-color artwork:

| Project | What it is strong at | Where this tool is different |
| --- | --- | --- |
| [Potrace](https://github.com/skyrpex/potrace) | Smooth vector paths from a black/white bitmap | Potrace is used here only after the icon foreground mask has been recovered. |
| [AutoTrace](https://github.com/autotrace/autotrace) | Classic bitmap-to-vector conversion with outline/centerline tracing, despeckling, color reduction, and many output formats | This tool focuses on one web output shape: inline SVG HTML for small UI icons. |
| [VTracer](https://github.com/visioncortex/vtracer) | Color raster-to-vector conversion for scans, graphics, photos, and pixel art | This tool is tuned for tiny icon crops where background removal is usually harder than curve fitting. |
| [ImageTracerJS](https://github.com/jankovicsandras/imagetracerjs) | Browser/Node image-to-SVG tracing with palette and preprocessing options | This tool does not try to vectorize every color layer. It tries to isolate one icon first. |
| Recent research such as [SAMVG](https://arxiv.org/abs/2311.05276), [StarVector](https://arxiv.org/abs/2312.11556), and [AmodalSVG](https://arxiv.org/abs/2604.10940) | General image-to-SVG generation, segmentation-assisted vectorization, or editable semantic layers | This tool is a lightweight local pipeline for cropped UI icons, not a general SVG generation model. |

The niche is noisy UI icon crops, especially icons taken from screenshots,
mockups, or AI-generated interface images where the foreground icon may sit on a
dark, textured, colorful, or patterned background. In those cases, directly
running a tracer often copies background texture into the SVG. Auto Icon
Vectorizer treats mask recovery as the main problem, then uses Potrace for the
final vector path.

Assumptions:

- the input is already cropped around one icon
- the icon foreground is mostly one visual color
- the icon can be outline, filled, or a same-color fill+stroke hybrid
- the background can be noisy, colorful, or textured, but it must still contain
  enough color or contrast evidence to separate it from the icon
- the output is visually clean Potrace path SVG, not hand-authored SVG made from
  editable circles, lines, rectangles, or text objects
- multicolor logos, full screenshots, text recognition, and object detection are
  outside the current scope

The production renderer is:

```text
auto-stroke-filled+potrace-default
```

It runs two learned mask branches and selects the one that reconstructs the crop
best:

```text
stroke / outline icons
  -> gated U-Net stroke mask
  -> Potrace
  -> inline SVG HTML

filled / silhouette / hybrid icons
  -> filled-silhouette U-Net mask
  -> Potrace
  -> inline SVG HTML
```

## Results

The examples below show the final output only: original crop on the left,
generated inline SVG on the right. The checkerboard means the SVG background is
transparent.

![results overview](examples/results-overview.png)

What this demonstrates:

- background removal on dark, light, noisy, and colored crops
- outline icons with connected strokes
- filled icons with holes/cutouts
- mixed fill+stroke icons
- automatic selection between the stroke and filled mask branches

Current regression status:

```text
6/6 examples pass
filled branch selected: 2
stroke branch selected: 4
```

## Install

Requirements:

- Python 3.9+
- Node.js + npm
- Python packages listed in `pyproject.toml`
- Node package `potrace`, installed into `auto_icon_vectorizer/runtime`

Local install:

```bash
git clone https://github.com/jaydenbarnescs-tech/auto-icon-vectorizer.git
cd auto-icon-vectorizer
python3 -m pip install -e .
python3 -m auto_icon_vectorizer.install_runtime
```

The Node install step is required because the final bitmap-to-SVG tracing call
uses the npm `potrace` package. The neural network checkpoints are included in
the repo; the large training feature cache is intentionally not included.

## CLI Usage

```bash
auto-icon-vectorizer path/to/icon-crop.png \
  --out-prefix examples/my-icon \
  --json examples/my-icon.json \
  --source-id my_icon \
  --class-name vector-icon
```

The command writes:

- `examples/my-icon.svg`
- `examples/my-icon.html`
- `examples/my-icon-source.png`
- `examples/my-icon-mask.png`
- `examples/my-icon-rendered.png`
- `examples/my-icon.json`

Run the packaged regression sheet:

```bash
auto-icon-vectorizer-regression
```

## Python API

```python
from pathlib import Path
from PIL import Image
from auto_icon_vectorizer import vectorize_icon_crop

crop = Image.open("icon-crop.png").convert("RGB")
result = vectorize_icon_crop(
    crop,
    source_id="feature_icon_001",
    class_name="vector-icon feature-icon",
    output_prefix=Path("out/feature_icon_001"),
    mask_mode="auto",
)

html = result["html"]  # <span ...><svg ...>...</svg></span>
svg = result["svg"]    # raw SVG only
diagnostics = result["diagnostics"]
```

The return value is designed for apps that generate or edit web pages:

```python
{
    "html": "...inline SVG HTML...",
    "svg": "...raw SVG...",
    "paths": [
        {"type": "potrace_path", "pathCount": 1, "renderer": "..."}
    ],
    "diagnostics": {
        "pipelineRenderer": "auto-stroke-filled+potrace-default",
        "renderer": "filled-silhouette-unet+potrace-default",
        "requestedMaskMode": "auto",
        "selectedMaskMode": "filled",
        "maskStrategy": "filled-silhouette-unet",
        "foregroundPixels": 5183,
        "strokeColor": "#c84a2b",
        "candidateScores": [...]
    }
}
```

## What The Algorithm Takes As Input

Input is an already-cropped raster icon image.

Good inputs:

- a 32-256 px crop around a single UI icon
- icon can be on dark, light, noisy, textured, or colorful AI-generated backgrounds
- icon foreground is mostly one visual color
- icon can be outline, filled, or a same-color fill+stroke hybrid

Bad inputs:

- a full screenshot instead of one cropped icon
- an icon crop containing several unrelated objects
- multicolor logos where each color needs to stay separate
- extremely small, blurred, or heavily occluded icons
- icon and background with almost identical color evidence

Output is HTML containing SVG, not just SVG. This makes it easy to place the
result directly into a generated page while still keeping the raw SVG available.

## How It Works

Detailed algorithm notes are in [docs/ALGORITHM.md](docs/ALGORITHM.md).

Short version:

1. Normalize the crop to 128 x 128 RGB.
2. Build many per-pixel evidence channels: RGB, HSV, Lab residuals, alpha-like
   chromatic evidence, spectral high-high evidence, local contrast, gradients,
   and coordinates.
3. Run the stroke gated U-Net branch.
4. Run the filled-silhouette gated U-Net branch.
5. Clean each mask using median filtering, component filtering, small-hole
   handling, and Potrace-specific preprocessing.
6. Estimate the icon stroke/fill color from masked pixels.
7. Trace each selected mask with Potrace.
8. Render the SVG back over an inpainted background estimate and score the
   reconstruction.
9. Select filled when it is plausible, materially larger than the stroke mask,
   and visually close or better; otherwise select stroke.
10. Return `<span data-vectorizer="..."><svg>...</svg></span>`.

## Papers And Methods Actually Used

The final production path is not SAM, not random forest, and not SVM-first.
Those were explored, but the shipped renderer is U-Net segmentation plus
Potrace tracing.

The papers and methods that directly shaped the final implementation:

- Peter Selinger, **Potrace: a polygon-based tracing algorithm** (2003). Used
  for mask-to-Bezier SVG tracing.
- Ronneberger, Fischer, Brox, **U-Net: Convolutional Networks for Biomedical
  Image Segmentation** (2015). Used as the architectural basis for pixel mask
  segmentation: encoder/decoder, skip-localization idea, strong synthetic data
  augmentation.
- Salehi, Erdogmus, Gholipour, **Tversky loss function for image segmentation**
  (2017). Used in the training loss family to handle foreground/background
  imbalance.
- Nobuyuki Otsu, **A Threshold Selection Method from Gray-Level Histograms**
  (1979). Used in evidence-channel thresholding and earlier mask baselines;
  retained as part of the evidence pipeline.
- Classical mathematical morphology and connected component filtering. Used for
  mask cleanup, tiny speck removal, and preserving intentional holes in filled
  icons.

Explored but not default:

- SVM pixel masks and SVM endpoint connection repair.
- Frangi vesselness and Fraz-style line/vessel evidence.
- GrabCut-style foreground rescue.
- SAM/SAM2 as an auxiliary signal.
- Random forest / XGBoost were discussed but intentionally not used in the
  shipped branch.

More detail and source links are in [docs/RESEARCH.md](docs/RESEARCH.md).

## Capabilities

Works well for:

- outline UI icons
- filled map pins, tags, stars, hearts, bookmarks, shields, etc.
- same-color hybrid icons with filled body plus stroke details
- simple holes/cutouts such as map-pin centers or tag holes
- noisy AI-generated backgrounds where the icon color is consistent
- returning transparent SVG paths without copying the background

Known weak points:

- true multicolor icons are collapsed to one estimated foreground color
- very thin strokes can still become slightly chunky because Potrace traces a
  binary mask boundary
- filled-only is not safe for all outline icons, so the auto selector keeps the
  stroke branch
- if an AI background contains icon-colored marks that touch or mimic the icon,
  the mask can over-include them
- this does not infer basic SVG shapes like "circle", "line", or
  "rounded rectangle"; it returns Potrace paths

See [docs/CAPABILITIES.md](docs/CAPABILITIES.md) for detailed case behavior.

## Repository Layout

```text
auto_icon_vectorizer/
  vectorize.py                         # public API + CLI
  regression.py                        # visual regression sheet generator
  install_runtime.py                   # npm install helper
  runtime/
    trace_icon_component.py             # mask cleanup, Potrace call, SVG normalization
    train_aux_fusion_icon_segmenter.py  # stroke gated U-Net architecture/features
    train_filled_silhouette_segmenter.py# filled silhouette model/features
    apply_svm_connections.py            # visual-diff utilities; SVM experiments are not default
    generate_spectral_evidence_bank.py  # evidence-map helpers
    nn-seg-results/
      best-gated-unet.pt
      best-filled-silhouette-unet.pt
examples/
  icon-vectorizer-regression.png
  hybrid-fill-stroke-eval-after-auto-threshold.png
  real-filled-vs-stroke-eval.png
docs/
  ALGORITHM.md
  RESEARCH.md
  CAPABILITIES.md
```

## License

MIT. See [LICENSE](LICENSE).
