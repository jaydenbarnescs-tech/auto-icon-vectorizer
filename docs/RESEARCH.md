# Research Notes

This document separates the methods that actually ended up in the default
renderer from methods that were explored and rejected or left as non-default
experiments.

## Production Methods

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

- Potrace traces boundaries. It does not infer semantic primitives like
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
