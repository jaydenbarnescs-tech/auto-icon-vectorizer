# Capabilities And Failure Modes

This page is written for someone deciding whether to use this tool in an app,
website, or design tool.

## What It Handles Well

### Outline Icons

Examples:

- cube outline
- check mark inside circle
- chat bubble with slash
- shopping cart line icon
- form/list line icon
- storefront line icon

Expected behavior:

- auto selector chooses `stroke`
- output is a transparent SVG path
- background is not copied

Why it works:

- the stroke gated U-Net was trained on thin-line foreground masks
- the filled branch is not forced when it does not add meaningful area

### Filled Icons

Examples:

- map marker
- tag
- shield
- star
- heart
- bookmark
- play button

Expected behavior:

- auto selector chooses `filled` when the filled mask is materially larger and
  plausible
- intentional holes are preserved where possible

Why it works:

- the filled-silhouette model uses color residuals, border color distance, edge
  evidence, and center-position evidence
- tiny pinholes are cleaned without automatically filling real icon cutouts

### Half Filled / Half Stroke Icons

Examples:

- filled map pin with route line
- filled tag with string stroke
- filled heart with ECG line
- filled star with orbit arc

Expected behavior:

- auto selector often chooses `filled`
- the filled branch usually captures same-color stroke details as part of the
  silhouette
- naive union is not used because it tends to add background fragments

### AI-Generated Patterned Backgrounds

The model is specifically designed for noisy AI backgrounds where a simple
black/white threshold fails.

Useful evidence:

- consistent icon color
- contrast from smooth local background
- border color distance
- Lab and RGB residuals
- alpha-like chromatic evidence

## What It Does Not Do

### It Does Not Detect Icons In A Full Screenshot

This tool does not search a full screenshot for icons. It expects an image
crop that already contains one icon.

### It Does Not Rebuild Icons As Editable Shapes

Output is Potrace paths, not:

- `<circle>`
- `<line>`
- `<rect>`
- named icon parts
- designer-editable source geometry

That is acceptable for visual reconstruction, but not ideal if you need a
designer-editable icon source.

### It Does Not Preserve Multiple Foreground Colors

The current SVG output uses one estimated foreground color. True multicolor
logos or icons where each color must stay separate are out of scope.

### It Does Not Reconstruct Missing Information

If the crop is too blurry, too small, or the foreground is indistinguishable
from the background, the model cannot recover details that are not present in
the pixels.

### It Does Not Guarantee Perfect Curves

Potrace traces the mask boundary. If the mask is chunky, the SVG can still look
chunky. The current cleanup reduces this but does not replace Potrace with a
centerline/stroke renderer.

## Known Failure Modes

### Icon-Colored Background Marks

If a background pattern has marks with the same color and local structure as
the icon, the mask can include them.

Mitigation:

- tighter icon crop
- better crop boundaries
- train more examples of that background family

### Very Thin Strokes

Very thin strokes can disappear in the filled branch. The auto selector keeps
the stroke branch for this reason.

Mitigation:

- keep `mask_mode="auto"`
- do not force `filled` globally

### Large Filled Icons With Internal Texture

If the icon body contains texture that looks like background, internal holes or
missing chunks can appear.

Mitigation:

- train filled examples with internal texture
- lower hole-fill aggressiveness only for cases where cutouts are expected

### Text-Like Icons

Small text inside icons is usually not preserved as text. It becomes paths or is
removed as noise.

Mitigation:

- OCR/text pipeline should handle text separately
- do not use this as a logo OCR system

## Recommended Integration Policy

Use this tool like this:

```text
app finds or receives an icon crop
    -> pass the crop to vectorize_icon_crop(crop, mask_mode="auto")
    -> use result["html"] in generated page
    -> keep original crop as fallback asset
    -> store diagnostics next to your icon record
```

Do not:

- pass the full page screenshot
- force filled-only
- union stroke and filled masks by default
- treat visual-diff score as ground truth

## Regression Cases Included

The included regression generator creates six cases:

- `hybrid_pin_route`: expected filled
- `hybrid_tag_string`: expected filled
- `hybrid_star_orbit`: accepts either branch if IoU is high enough
- `outline_cube`: expected stroke
- `outline_check_circle`: expected stroke
- `outline_chat_slash`: expected stroke

Run:

```bash
auto-icon-vectorizer-regression
```

The output sheet and JSON are written to `examples/regression-output/`.
