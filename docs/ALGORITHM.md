# Algorithm

This document describes the default path in `auto-stroke-filled+potrace-default`.

The central design decision is that tracing is only the final step. The hard
problem is producing a foreground mask that contains the icon and not the
background. Once the mask is good, Potrace can convert the mask boundary into
smooth Bezier paths.

## Input And Output Contract

Input:

- one cropped raster icon image
- accepted as a `PIL.Image.Image`
- any input width/height that Pillow can load
- the crop is converted to RGB and letterboxed to 128 x 128 pixels internally
- it does not search a full screenshot for icons

Output:

- `html`: a `<span>` containing an inline `<svg>`
- `svg`: raw SVG only
- `paths`: simple metadata about the generated Potrace paths
- `diagnostics`: selected branch, mask stats, stroke color, Potrace options,
  visual-diff scores, and optional artifact paths

Raw SVG is the default CLI output. The returned HTML wrapper is available for
web apps, site builders, and design tools that want one DOM-ready string with
CSS and metadata hooks.

## Branch 1: Stroke / Outline Segmentation

The stroke branch is used for line icons and outline icons.

Runtime name:

```text
nn-gated-unet+potrace-default
```

Mask strategy:

```text
nn-gated-unet
```

Feature channels:

- RGB channels
- alpha-like constant-color evidence
- warm and purple alpha heat bands
- warm/purple joined evidence
- spectral high-high evidence
- best spectral channel and thresholded version
- alpha flood evidence
- alpha gradient magnitude
- local dark and local light contrast
- HSV saturation and value
- Lab residual magnitude
- Lab a/b residual magnitude
- RGB residual magnitude
- normalized x/y coordinates

Model:

- compact U-Net-like encoder/decoder
- split main/aux inputs
- learned gate fuses main image channels and auxiliary evidence channels
- segmentation head plus boundary head

Why this branch exists:

Filled silhouette segmentation is not safe for all outline icons. On some cream
or gold line icons, a filled-silhouette model can decide that there is no
meaningful filled object and return an empty or nearly invisible mask. The
stroke branch preserves those pure outlines.

## Branch 2: Filled / Silhouette Segmentation

The filled branch is used for filled icons and mixed fill+stroke icons.

Runtime name:

```text
filled-silhouette-unet+potrace-default
```

Mask strategy:

```text
filled-silhouette-unet
```

Feature channels:

- RGB channels
- grayscale
- hue encoded as sine/cosine
- saturation and value
- Lab residual from smoothed background
- Lab a/b residual
- RGB residual
- distance from border median color in Lab and RGB
- local dark and local light contrast
- edge magnitude
- normalized center distance
- normalized x/y coordinates

Post-processing:

- probability hysteresis
- median cleanup
- connected component filtering
- tiny pinhole fill
- intentional hole preservation

The hole preservation is important. Earlier cleanup filled all small internal
holes, which broke map-pin centers and tag holes. The current cleanup only fills
tiny raster pinholes and leaves real cutouts intact.

## Auto Selector

The adapter traces both branches when possible. Then it decides which SVG to
return.

Selection signals:

- foreground area of stroke mask
- foreground area of filled mask
- filled/stroke area ratio
- visual reconstruction score after rendering SVG back onto an inpainted
  background estimate

Current rule:

```text
if filled area is plausible
and filled area >= max(0.115, stroke area * 1.25)
and filled visual score <= stroke visual score + 0.012:
    select filled
else:
    select stroke
```

This rule came from testing:

- pure outline icons should usually stay stroke
- filled and hybrid icons should switch to filled when the filled branch adds
  a real body/silhouette
- naive union of stroke and filled masks is worse because it often adds
  background fragments

## Potrace Stage

Both branches produce binary masks. The selected mask is preprocessed for
Potrace:

- median filtering
- small component removal
- tiny white-hole filling
- thickness estimation
- option selection based on mask thickness

Potrace options are conservative:

```python
{
    "turdSize": 2,
    "optTolerance": 0.18 or 0.26,
    "alphaMax": 0.95 or 1.12,
}
```

The goal is to smooth jagged mask boundaries without erasing icon features.

## Color Estimation

The mask defines the foreground pixels. The algorithm estimates a single
foreground color from those pixels and injects it into the SVG paths.

This is why the SVG does not copy the background. The background is only used
for scoring and debugging.

The returned SVG contains foreground paths only. There is no background
rectangle, so the icon background is transparent when rendered by a browser or
SVG renderer.

## Visual-Diff Scoring

The selector renders the candidate SVG onto an inpainted estimate of the
original background, then compares that composite against the original crop.

Metrics:

- RGB mean absolute error
- RGB RMSE
- edge difference
- high-difference pixel ratio
- dark pixel ratio delta

The current priority score is a weighted mix of those values. This is useful
for choosing between candidates, but it is not a perfect truth metric. Thin
stroke failures can look numerically cheap because the missing pixels occupy a
small part of the crop. That is why routing rules also consider mask area.

## Why Not Filled-Only

Filled-only looked strong on hybrid icons, but it failed on some pure outline
icons. The regression suite includes this behavior:

- hybrid pin route -> filled
- hybrid tag string -> filled
- pure outline cube/check/chat icons -> stroke

The default behavior is therefore not "always filled". It is an automatic
selector that keeps both branches.

## Why Not Union

Union means:

```text
final mask = stroke mask OR filled mask
```

It was tested and rejected as the default. It often fattens the icon or adds
background artifacts from the stroke branch. The filled branch alone performed
better on same-color filled/hybrid icons.
