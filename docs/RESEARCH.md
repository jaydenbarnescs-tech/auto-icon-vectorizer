# Research Notes

This document separates the methods that actually ended up in the default
renderer from methods that were explored and rejected or left as non-default
experiments.

## Existing Landscape And The Gap

As of May 2026, there are strong tools and papers for raster-to-vector
conversion, but they mostly optimize for different inputs than this project.
The gap this project fills is: **recover a style-matched, blurry, mostly
single-color UI icon from an AI-generated screenshot/mockup and return a
transparent SVG/HTML asset that can be reused on a web page.**

That is not the same as tracing a clean logo, vectorizing a color photograph,
extracting all objects from a scene, or generating semantic SVG code from
scratch.

### Classic Binary Tracing

Representative tools:

- [Potrace](https://potrace.sourceforge.net/)
- Potrace integrations in tools such as Inkscape
- TypeScript/JS ports and wrappers around Potrace

What already existed:

- Fast, deterministic tracing from a black/white bitmap into smooth paths.
- Good curve fitting once the input mask is already clean.
- Mature SVG/PDF/EPS-style output paths.

What was missing for this project:

- Potrace does not solve foreground extraction from a messy AI-generated crop.
- If the bitmap contains background texture, Potrace faithfully traces that
  texture too.
- The hard part for our target input is not curve fitting; it is deciding which
  pixels are the icon.

This project therefore uses Potrace as the final renderer, not as the full
solution.

### Classic And Open-Source Raster Vectorizers

Representative tools:

- [AutoTrace](https://github.com/autotrace/autotrace)
- [ImageTracerJS](https://github.com/jankovicsandras/imagetracerjs)
- [image-tracer-ts](https://github.com/mringler/image-tracer-ts)
- [VTracer](https://github.com/visioncortex/vtracer)

What already existed:

- AutoTrace covers classic bitmap-to-vector conversion, including outline and
  centerline tracing, color reduction, despeckling, and multiple output formats.
- ImageTracerJS and related browser/Node packages provide configurable
  image-to-SVG tracing for simple images and palette-style conversion.
- VTracer is a strong modern color raster-to-vector converter. Its README
  describes it as a tool for JPG/PNG to SVG and specifically notes that it can
  handle colored high-resolution scans where Potrace expects binarized input.

What was missing for this project:

- These tools are general vectorizers. They do not know that "the icon" is the
  foreground object to preserve and "the generated UI background" is disposable.
- Color-layer vectorizers can preserve or trace background layers when the
  desired output is only a transparent icon.
- For tiny blurry UI icon crops, over-vectorizing background detail is often
  worse than losing it.
- They generally expose SVG output, not an icon-specific web integration
  contract with `svg`, `html`, diagnostics, source ids, mask previews, and
  branch-selection metadata.

The gap is not "convert image to SVG". The gap is "convert the intended icon
inside this small AI-generated crop to SVG without copying the generated
background".

### Design Tools And Stock Icon Replacement

Representative options:

- Use an icon library such as Material Icons, Lucide, Font Awesome, Heroicons,
  or similar.
- Use design software image tracing.
- Ask an image model to redraw or enhance the icon.

What already existed:

- Icon libraries give clean, hand-authored SVG.
- Design tools can trace images interactively.
- Image models can create new icon-like raster assets.

What was missing for this project:

- A stock icon may not match the AI-generated UI's exact visual language,
  stroke weight, softness, color, or composition.
- A second image-generation pass costs time/tokens, is less reproducible, and
  still produces a raster image unless another vectorization step follows.
- Manual design-tool tracing is not a deterministic no-human interface for an
  automated UI generation pipeline.

Manual tracing tools are not the problem when a human designer is in the loop.
The problem is the interface. Interactive tracing assumes someone can inspect
the crop, choose a threshold or preset, decide whether background fragments were
copied, reconnect strokes, smooth bad corners, and export the final result. An
automated website generator needs the opposite: a stable crop-in, SVG/HTML-out
function that can run repeatedly without visual inspection.

On blurry AI-generated icon crops, the failure mode is usually upstream of curve
fitting:

- direct thresholding can merge icon pixels with similarly bright or dark
  background regions
- color tracing can preserve generated background layers that should be
  discarded
- edge tracing finds background edges as confidently as icon edges
- JPEG/PNG blur and antialiasing turn one stroke into a fuzzy band that needs
  foreground/background reasoning before tracing

The README visual sheet
[`examples/tracing-alone-failure-modes.png`](../examples/tracing-alone-failure-modes.png)
shows these failure modes on generated blurry icon crops. This is why this
project uses mask recovery first and Potrace second.

If a clean source icon already exists and style matching does not matter, that
source icon is better. This project is for the case where the generated icon
itself is the asset worth preserving.

### Recent Image-to-SVG Research

Representative directions:

- [SAMVG](https://arxiv.org/abs/2311.05276): segmentation-assisted multi-stage
  image vectorization using Segment Anything.
- [StarVector](https://arxiv.org/abs/2312.11556): SVG code generation from
  image/text using a vision-language model.
- [AmodalSVG](https://arxiv.org/abs/2604.10940): semantic layer peeling and
  amodal vectorization for editable object layers.
- [LIVE](https://arxiv.org/abs/2206.04655): layer-wise image vectorization.
- Broader survey work such as [Image Vectorization: a Review](https://arxiv.org/abs/2306.06441).

What already existed:

- Research systems increasingly address semantic layers, primitive generation,
  differentiable rendering, segmentation-assisted vectorization, and direct SVG
  code generation.
- These are closer to "understand the image and produce structured vector
  graphics".

What was missing for this project:

- The target runtime here is small, local, inspectable, and task-specific.
- The input is not a general scene or artwork; it is one cropped UI icon.
- The output does not need semantic SVG primitives. It needs a reliable,
  transparent, scalable web asset.
- Foundation segmentation or SVG-code generation models add runtime complexity
  that is hard to justify for small icon crops.

The project therefore uses a compact learned segmentation model plus classical
tracing instead of a large general-purpose image-to-SVG model.

### What This Project Adds

This project's contribution is the combination of these choices for a specific
under-covered case:

- target input: blurry, noisy, AI-generated UI icon crops
- target output: transparent SVG plus optional inline SVG HTML
- main technical bet: mask recovery matters more than curve fitting
- segmentation: two compact U-Net-style branches, one for stroke/outline icons
  and one for filled/silhouette icons
- evidence: RGB, HSV, Lab residuals, local contrast, chromatic evidence,
  gradients, coordinates, and branch-specific features
- cleanup: conservative morphology and connected components so Potrace does not
  trace obvious background fragments
- selection: render-back scoring chooses between stroke and filled candidates
- integration: CLI, Python API, diagnostics, mask previews, source ids, and a
  doctor command for fresh-clone setup

This is why the project can be considered state-of-the-art for its narrow
practical niche: not because it invents a new universal vectorization theory,
but because it packages the strongest useful pieces for the exact problem of
turning AI-generated UI icon pixels into reusable web SVG.

During development, we did a broad AI-assisted search across classic tracing
tools, color vectorizers, interactive design workflows, segmentation-based
approaches, and recent image-to-SVG research. We did not find an open-source
local tool whose primary contract was this exact AI-generated UI icon-crop
problem. That is why this project exists.

## Default Methods

### Potrace

Source:

- Peter Selinger, **Potrace: a polygon-based tracing algorithm**, 2003.
- PDF: https://www.mathstat.dal.ca/~selinger/potrace/potrace.pdf

How it is used:

- The neural network outputs a binary foreground mask.
- The mask is converted to black-on-white bitmap form.
- The Node `potrace` package traces mask boundaries into SVG Bezier paths.
- We tune `turdSize`, `optTolerance`, and `alphaMax` based on mask thickness.

Why it stayed:

- Once the mask is good, Potrace produces compact smooth SVG paths.
- It is deterministic and fast enough for a UI extraction pipeline.
- It avoids hand-rolling curve fitting.

Important limitation:

- Potrace traces boundaries. It does not infer basic SVG shapes like
  `circle`, `line`, or `rect`, and it does not know what a "check mark" or
  "map pin" is.

### U-Net-Style Segmentation

Source:

- Olaf Ronneberger, Philipp Fischer, Thomas Brox, **U-Net: Convolutional
  Networks for Biomedical Image Segmentation**, 2015.
- arXiv: https://arxiv.org/abs/1505.04597

How it is used:

- The stroke model and filled model are compact U-Net-like encoder/decoder
  networks.
- Skip-style localization is important because icon masks need pixel-level
  precision.
- The models are trained on synthetic/stress cases with heavy background and
  color variation.

Why it stayed:

- Earlier threshold and SVM-first methods were strong on black/white cases but
  failed on colorful patterned AI backgrounds.
- U-Net-like segmentation let the model use multiple evidence channels at once
  instead of forcing one brittle mask heuristic.

### Tversky / Dice-Style Losses

Source:

- Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Ali Gholipour, **Tversky loss
  function for image segmentation using 3D fully convolutional deep networks**,
  2017.
- arXiv: https://arxiv.org/abs/1706.05721

How it is used:

- The training losses combine BCE, Dice, Tversky-style imbalance handling, and
  boundary supervision.
- This helps because icon foreground is usually a minority of the crop.

Why it stayed:

- Plain pixel accuracy would reward predicting background too often.
- Tversky/Dice-style terms push the model toward recall/precision tradeoffs
  that are better for small foreground masks.

### Otsu Thresholding

Source:

- Nobuyuki Otsu, **A Threshold Selection Method from Gray-Level Histograms**,
  1979.
- DOI: https://doi.org/10.1109/TSMC.1979.4310076

How it is used:

- Used in the evidence stack and in earlier fixed-mask baselines.
- Still useful for converting continuous evidence channels into binary helper
  channels.

Why it is not enough alone:

- Otsu assumes useful histogram separation. Patterned and colorful backgrounds
  often break that assumption.

### Morphology And Connected Components

Source category:

- Classical binary image morphology and connected component analysis.

How it is used:

- Remove tiny fragments.
- Fill only tiny pinholes.
- Preserve real cutouts such as map-pin centers and tag holes.
- Prevent Potrace from tracing obvious raster specks.

Why it stayed:

- Neural masks still need deterministic cleanup before tracing.
- The cleanup is deliberately conservative because over-cleaning destroys icon
  holes.

## Explored But Not Default

### SVM

What was tried:

- SVM endpoint connection repair.
- SVM pixel classification using color/evidence channels.

Why it did not stay as the main path:

- SVM was useful for thinking about features and line connectivity, but it did
  not handle the full colorful-background mask problem as well as the learned
  U-Net branches.
- The final renderer does not use random forest or XGBoost either.

### Frangi Vesselness

Source:

- Alejandro F. Frangi et al., **Multiscale Vessel Enhancement Filtering**,
  1998.
- DOI: https://doi.org/10.1007/BFb0056195

What was tried:

- Treat strokes like vessels/curves and enhance line-like structures.

Why it did not stay as the main path:

- It can help with thin line evidence, but UI icons are not always vessel-like.
- Filled icons and noisy patterned backgrounds created too many false positives.

### Fraz-Style Retinal Vessel Classification

Source:

- M. M. Fraz et al., **An ensemble classification-based approach applied to
  retinal blood vessel segmentation**, 2012.
- PubMed: https://pubmed.ncbi.nlm.nih.gov/22736688/

What was tried:

- Use line/vessel segmentation ideas as auxiliary evidence before the SVM/NN
  mask.

Why it did not stay as the main path:

- It is conceptually relevant for line-like strokes, but the icon problem also
  contains filled silhouettes, holes, and colorful AI backgrounds.

### GrabCut

Source:

- Rother, Kolmogorov, Blake, **GrabCut: Interactive Foreground Extraction using
  Iterated Graph Cuts**, 2004.
- PDF: https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf

What was tried:

- Use foreground/background color modeling as a rescue idea around evidence
  masks.

Why it did not stay as the main path:

- It is useful when the foreground/background color model is clean. In our icon
  crops, background patterns can share colors with the icon and confuse the
  graph-cut objective.

### SAM / SAM2

What was considered:

- Use a segmentation foundation model as an auxiliary mask or prompt with
  "remove everything but the icon".

Why it did not ship:

- The required output is a precise icon mask for small UI crops, not just an
  object region.
- Local runtime complexity is much higher.
- The task is narrow enough that a small dedicated model is faster and easier to
  inspect.

## Open-Source Components Used

- Potrace npm package: called from Node for bitmap-to-SVG tracing.
- PyTorch: neural network inference/training.
- OpenCV: color conversion, Sobel gradients, morphology, connected components,
  distance transforms, inpainting.
- scikit-image: thresholding and exploratory filters.
- CairoSVG: render SVG back to PNG for diagnostics and visual-diff scoring.

## Practical Conclusion

The final working recipe is:

```text
good learned mask -> conservative cleanup -> Potrace -> render-back scoring -> HTML/SVG output
```

The mask is the product. Potrace is the renderer. The auto selector exists
because one model does not cover all icon families safely.
