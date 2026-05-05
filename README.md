# Auto Icon Vectorizer - AI UI Icon Crop to SVG

Recover blurry AI-generated UI icon crops as clean SVG and inline SVG HTML.

Auto Icon Vectorizer is an image-to-SVG tool for small UI icons found inside
AI-generated interface images, screenshots, and mockups. It takes one cropped
raster icon, removes the background, traces the recovered foreground mask, and
returns both raw SVG and an HTML wrapper containing that same inline SVG.

## Why This Exists

This project is mainly for AI-generated UI images, screenshot-to-code
workflows, and blurry icon crops that do not have a clean source SVG.

Image models can generate UI mockups where the icons match the page's visual
style, color, lighting, and theme. The problem is that those icons usually only
exist as small blurry raster pixels inside the generated image. Replacing them
with a stock icon from a library can be cleaner, but it can also change the
look of the design. Asking an image model to redraw every icon is slower, more
expensive, less reproducible, and still leaves the page with raster image
assets.

Auto Icon Vectorizer takes the icon that already exists in the generated UI
image and turns it into a lightweight SVG. That gives the web page a scalable,
fast-loading asset while preserving more of the style that was present in the
original generated design.

One concrete workflow looks like this:

1. An image model generates a website or app mockup.
2. A screenshot-to-code, website-rebuilder, or UI extraction step finds a small
   crop around one icon.
3. That crop is passed to Auto Icon Vectorizer.
4. The returned SVG or inline SVG HTML is inserted back into the final generated
   website.

![AI website icon workflow](examples/ai-website-icon-workflow.png)

The important point is that this package does not need to understand the whole
website. It only needs the cropped icon. The surrounding system can handle page
layout, text, component structure, and icon detection; Auto Icon Vectorizer
handles the part where a low-quality generated icon crop becomes a transparent,
scalable web asset.

![AI website HTML integration](examples/ai-website-html-integration.png)

If a clean source icon already exists, use that source icon. This tool is most
useful when the icon only exists as pixels in a generated image, screenshot, or
mockup and the goal is to recover a usable web asset from those pixels.

The missing piece is not another general raster-to-vector tracer. Existing tools
already handle clean bitmaps, scans, logos, photos, pixel art, and full-color
artwork. The under-covered case is much narrower: a tiny blurry icon crop from
an AI-generated UI image where the icon style is valuable, but the background is
noisy and the icon only exists as pixels.

In that case, directly running a tracer often copies background texture into the
SVG. Replacing the icon with a library icon can lose the generated design's
style. Regenerating the icon with an image model is heavier, less reproducible,
and still leaves a raster asset. Auto Icon Vectorizer fills this gap by treating
foreground mask recovery as the main problem, then using Potrace only after the
icon has been separated from the background.

The detailed landscape review is in [docs/RESEARCH.md](docs/RESEARCH.md). It
covers classic tracers, color vectorizers, design tools, recent image-to-SVG
research, and why this project focuses on AI-generated UI icon crops.

![pipeline diagram](examples/pipeline-diagram.png)

Use this project when:

- an app already has a small crop around an icon from an AI-generated UI image,
  screenshot, or mockup
- the goal is clean HTML/SVG that can be inserted into a web page
- the icon should be separated from a noisy or AI-generated background
- the icon is a single-color foreground shape, including outline, filled, or
  same-color fill+stroke icons

Choose a different tool when:

- a clean SVG, font icon, or icon-library match already exists and style drift
  is acceptable
- the input is a full photo, illustration, scan, or complex logo
- the desired output is a fully editable SVG rebuilt from circles, lines,
  polygons, text, and named layers
- every foreground color in a multicolor logo must remain separate
- the task is to find icons inside a full screenshot

Assumptions:

- the input is already cropped around one icon
- the icon foreground is one visual color; the model recovers one foreground
  mask and the SVG output uses one fill color
- the icon can be outline, filled, or a same-color fill+stroke hybrid
- the background can be noisy, colorful, or textured, but it must still contain
  enough color or contrast evidence to separate it from the icon
- the output is visually clean Potrace path SVG, not hand-authored SVG made from
  editable circles, lines, rectangles, or text objects
- multicolor logos, full screenshots, text recognition, and object detection are
  outside the current scope

Default pipeline:

```text
auto-stroke-filled+potrace-default
```

It runs two learned mask branches and selects the one that reconstructs the crop
best:

```text
stroke / outline icons
  -> gated U-Net stroke mask
  -> Potrace
  -> SVG + optional inline SVG HTML

filled / silhouette / hybrid icons
  -> filled-silhouette U-Net mask
  -> Potrace
  -> SVG + optional inline SVG HTML
```

## Output Contract

The CLI prints raw SVG by default because SVG is the portable asset format. The
Python API always returns both:

- `result["svg"]`: raw SVG, best for saving an asset or passing to another
  vector tool
- `result["html"]`: a `<span>` wrapper containing the same SVG, best when code
  wants one DOM-ready string with class names, source id, renderer metadata,
  title, or aria label

![output contract](examples/output-contract.png)

## Transparent Background

The returned SVG is transparent by design. It contains foreground path data only
and does not include a background rectangle, so it can be placed over any page
background, card, button, or CSS surface. Visibility still depends on the icon
color having enough contrast with the target background.

![transparent SVG backgrounds](examples/transparent-backgrounds.png)

## Icon Color

By default, the vectorizer estimates the icon's original foreground color from
the recovered mask and writes that color into the SVG paths. You can override
that output color at the CLI or API level without changing the segmentation
model or Potrace trace.

```bash
auto-icon-vectorizer path/to/icon-crop.png --icon-color '#111827' > icon.svg
auto-icon-vectorizer path/to/icon-crop.png --stdout html --icon-color currentColor
```

Accepted color values are hex colors, CSS color names, `currentColor`, and
simple CSS variables such as `var(--icon-color)`. Use `currentColor` when the
HTML wrapper should inherit color from CSS:

```html
<span class="feature-icon" style="color:#2563eb">
  <svg ...><path fill="currentColor" ... /></svg>
</span>
```

The output still contains one foreground color. This project does not split a
multicolor logo into separate SVG layers.

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
- native Cairo library for SVG rendering through CairoSVG
- Python packages listed in `pyproject.toml`
- Node package `potrace`, installed into `auto_icon_vectorizer/runtime`

Local install:

```bash
git clone https://github.com/jaydenbarnescs-tech/auto-icon-vectorizer.git
cd auto-icon-vectorizer
python3 -m pip install -e .
python3 -m auto_icon_vectorizer.install_runtime
python3 -m auto_icon_vectorizer.doctor
```

The Node install step is required because the final bitmap-to-SVG tracing call
uses the npm `potrace` package. The neural network checkpoints are included in
the repo; the large training feature caches and original local training corpora
are intentionally not included.

For exact details on included files, omitted generated artifacts, and retraining
from a public synthetic corpus, see [docs/RUNTIME_ASSETS.md](docs/RUNTIME_ASSETS.md).

## CLI Usage

```bash
auto-icon-vectorizer path/to/icon-crop.png \
  --out-prefix examples/my-icon \
  --json examples/my-icon.json \
  --source-id my_icon \
  --class-name vector-icon \
  --icon-color '#111827'
```

By default, the command prints SVG to stdout.

The command writes:

- `examples/my-icon.svg`
- `examples/my-icon-source.png`
- `examples/my-icon-mask.png`
- `examples/my-icon-rendered.png`
- `examples/my-icon.json`

To print the HTML wrapper instead of raw SVG:

```bash
auto-icon-vectorizer path/to/icon-crop.png --stdout html --icon-color currentColor
```

To also write a standalone HTML preview next to the SVG artifacts:

```bash
auto-icon-vectorizer path/to/icon-crop.png \
  --out-prefix examples/my-icon \
  --write-html
```

To print the full diagnostic JSON to stdout:

```bash
auto-icon-vectorizer path/to/icon-crop.png --stdout json
```

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
    icon_color="#111827",
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
        "outputColor": "#111827",
        "colorOverride": "#111827",
        "candidateScores": [...]
    }
}
```

## What The Algorithm Takes As Input

Input is an already-cropped raster icon image. It does not need to be 128 x 128.
The public API accepts any image dimensions that Pillow can load. Internally,
the crop is converted to RGB and letterboxed onto a 128 x 128 model canvas so
the neural networks and Potrace step operate on a consistent coordinate system.
The returned SVG has a `viewBox="0 0 128 128"` and can be scaled with CSS or
normal SVG attributes.

Good inputs:

- a crop around a single UI icon, from small UI captures to larger source crops
- icon can be on dark, light, noisy, textured, or colorful AI-generated backgrounds
- icon foreground is one visual color
- icon can be outline, filled, or a same-color fill+stroke hybrid

Bad inputs:

- a full screenshot instead of one cropped icon
- an icon crop containing several unrelated objects
- multicolor logos where each color needs to stay separate
- extremely small, blurred, or heavily occluded icons
- icon and background with almost identical color evidence

Output includes both raw SVG and HTML containing that same SVG. The CLI defaults
to raw SVG output; the HTML wrapper is available for direct web insertion.

## How It Works

Detailed algorithm notes are in [docs/ALGORITHM.md](docs/ALGORITHM.md).

Short version:

1. Accept an arbitrary-size crop, convert it to RGB, and letterbox it to the
   internal 128 x 128 model canvas.
2. Build many per-pixel evidence channels: RGB, HSV, Lab residuals, alpha-like
   chromatic evidence, spectral high-high evidence, local contrast, gradients,
   and coordinates.
3. Run the stroke gated U-Net branch.
4. Run the filled-silhouette gated U-Net branch.
5. Clean each mask using median filtering, component filtering, small-hole
   handling, and Potrace-specific preprocessing.
6. Estimate the icon stroke/fill color from masked pixels, or apply the caller's
   output color override.
7. Trace each selected mask with Potrace.
8. Render the SVG back over an inpainted background estimate and score the
   reconstruction.
9. Select filled when it is plausible, materially larger than the stroke mask,
   and visually close or better; otherwise select stroke.
10. Return raw SVG plus an optional `<span data-vectorizer="..."><svg>...</svg></span>`
    HTML wrapper.

## Technical Approach

The implementation combines learned foreground segmentation with classical
vector tracing:

- **U-Net-style segmentation** recovers a foreground mask from RGB, color-space,
  contrast, residual, gradient, coordinate, and chromatic evidence channels.
- **Separate stroke and filled branches** handle outline icons and filled icons
  differently, then an automatic selector chooses the cleaner reconstruction.
- **Tversky-style training loss** helps with the common imbalance between small
  icon foreground pixels and larger background areas.
- **Otsu thresholding, morphology, and connected components** clean the mask,
  remove small background specks, and preserve real holes such as map-pin
  centers or tag cutouts.
- **Potrace** converts the final binary mask into smooth SVG paths.
- **Color estimation** samples the recovered foreground so the SVG keeps the
  icon's original visual color instead of defaulting to black. Callers can
  override this single output color with `--icon-color` or `icon_color=...`.

## Training Data Shape

The neural networks are trained as foreground/background segmenters, not
multicolor vectorizers. The training target is a binary alpha mask: icon pixels
versus background pixels. The color of the icon is used as evidence for
separation, but the model is not trained to preserve separate red, blue, green,
or gradient regions inside one icon.

The public stroke-corpus generator draws each outline icon with one foreground
color over varied dark, light, striped, dotted, gradient, and noisy backgrounds.
The filled-silhouette training script generates filled shapes, cutouts, and
same-color hybrid-style silhouettes using one foreground color per icon. This
matches the shipped output contract: one recovered mask traced into transparent
SVG paths with one fill color.

When training your own checkpoint, prepare crops and truth masks with the same
contract:

- one cropped icon per image
- one dominant foreground color for the icon
- binary alpha truth mask marking all foreground pixels
- varied backgrounds that represent the screenshots or AI-generated UI images
  where the model will be used

Multicolor logos can still be converted into one-color silhouettes, but this is
a lossy fallback rather than the intended use case.

This is a practical hybrid pipeline. It is not intended to replace general
image vectorizers, full-scene segmentation models, or SVG-code foundation
models. It is intended for the narrower case where the input is already a small
icon crop and the main challenge is separating the icon from a messy background.

More detail and source links are in [docs/RESEARCH.md](docs/RESEARCH.md).

## Capabilities

Works well for:

- outline UI icons
- filled map pins, tags, stars, hearts, bookmarks, shields, etc.
- same-color hybrid icons with filled body plus stroke details
- simple holes/cutouts such as map-pin centers or tag holes
- noisy AI-generated backgrounds where the icon color is consistent
- returning transparent SVG paths without copying the background
- overriding the final SVG/HTML color with a CLI/API argument

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
  doctor.py                            # runtime readiness checker
  regression.py                        # visual regression sheet generator
  install_runtime.py                   # npm install helper
  runtime/
    trace_icon_component.py             # mask cleanup, Potrace call, SVG normalization
    train_aux_fusion_icon_segmenter.py  # stroke gated U-Net architecture/features
    train_filled_silhouette_segmenter.py# filled silhouette model/features
    apply_svm_connections.py            # visual-diff and mask utility helpers
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
  RUNTIME_ASSETS.md
  RESEARCH.md
  CAPABILITIES.md
```

## License

MIT. See [LICENSE](LICENSE).
