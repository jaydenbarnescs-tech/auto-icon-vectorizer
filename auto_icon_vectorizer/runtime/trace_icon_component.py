"""Mask-candidate tracing component for raster icons.

Input:
    A raster icon crop as a PIL RGB image.

Output:
    HTML with an inline SVG plus diagnostics/artifacts.

The important change from the earlier SVM-only prototype is that the mask is not
assumed. We generate multiple plausible foreground masks, trace each with
Potrace, render the SVG back over an inpainted background estimate, and select
the candidate that best reconstructs the source crop under the existing visual
diff metric.
"""

from __future__ import annotations

import ctypes.util
import io
import json
import math
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import frangi, threshold_otsu, threshold_sauvola
from skimage.morphology import remove_small_objects, skeletonize

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / ".vendor" / "python"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

from apply_svm_connections import diff_heatmap, score_pair


SIZE = 128
PIXEL_SVM_MODEL = ROOT / "svm-pixel-mask-model" / "pixel_stroke_svm.joblib"
NN_GATED_UNET_MODEL = ROOT / "nn-seg-results" / "best-gated-unet.pt"
_PIXEL_SVM_MODEL_CACHE: Any | None = None
_NN_GATED_UNET_CACHE: Any | None = None


@dataclass
class TraceCandidate:
    name: str
    mask: np.ndarray
    svg: str
    html: str
    rendered: Image.Image
    inpainted_background: Image.Image
    composite: Image.Image
    diff_overlay: Image.Image
    metrics: Dict[str, float]
    selection_score: float
    mask_quality: Dict[str, float]
    stroke_color: str


@dataclass(frozen=True)
class PotraceVariant:
    name: str
    mask: np.ndarray
    options: Dict[str, Any]
    penalty: float


def trace_icon_image(
    image: Image.Image,
    output_prefix: Path | None = None,
    size: int = SIZE,
    max_masks: int = 10,
) -> Dict[str, Any]:
    """Trace an icon crop and return best HTML with artifacts.

    The returned HTML is only the icon SVG wrapped in a span. Background is used
    for scoring only; it is not included in the SVG output.
    """

    source = canonical_image(image.convert("RGB"), size)
    masks = mask_candidates(source, max_masks=max_masks)
    candidates: List[TraceCandidate] = []

    for mask_name, mask in masks:
        try:
            candidate = trace_mask_candidate(source, mask_name, mask, size)
        except Exception:
            continue
        candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No mask candidate produced a traceable SVG")

    candidates.sort(key=lambda candidate: candidate.selection_score)
    best = candidates[0]

    artifacts = {}
    if output_prefix is not None:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        source_path = str(output_prefix) + "-source.png"
        source.save(source_path)
        artifacts["source"] = source_path
        for index, candidate in enumerate(candidates, start=1):
            stem = f"{output_prefix}-{index:02d}-{candidate.name}"
            mask_path = stem + "-mask.png"
            rendered_path = stem + "-rendered.png"
            background_path = stem + "-background.png"
            composite_path = stem + "-composite.png"
            diff_path = stem + "-diff.png"
            svg_path = stem + ".svg"
            html_path = stem + ".html"
            mask_debug_image(candidate.mask).save(mask_path)
            candidate.rendered.save(rendered_path)
            candidate.inpainted_background.save(background_path)
            candidate.composite.save(composite_path)
            candidate.diff_overlay.save(diff_path)
            Path(svg_path).write_text(candidate.svg, encoding="utf-8")
            Path(html_path).write_text(standalone_html(candidate.html), encoding="utf-8")
            if candidate is best:
                artifacts.update(
                    {
                        "bestMask": mask_path,
                        "bestRendered": rendered_path,
                        "bestBackground": background_path,
                        "bestComposite": composite_path,
                        "bestDiffOverlay": diff_path,
                        "bestSvg": svg_path,
                        "bestHtml": html_path,
                    }
                )

    return {
        "html": best.html,
        "svg": best.svg,
        "bestMask": best.name,
        "bestMetrics": best.metrics,
        "bestStrokeColor": best.stroke_color,
        "candidateCount": len(candidates),
        "candidates": [
            {
                "name": candidate.name,
                "metrics": candidate.metrics,
                "selectionScore": round(candidate.selection_score, 5),
                "maskQuality": candidate.mask_quality,
                "strokeColor": candidate.stroke_color,
                "pathCount": candidate.svg.count("<path"),
            }
            for candidate in candidates
        ],
        "artifacts": artifacts,
    }


def trace_mask_candidate(source: Image.Image, name: str, mask: np.ndarray, size: int) -> TraceCandidate:
    stroke_color = estimate_stroke_color(source, mask)
    candidates: list[TraceCandidate] = []
    errors: list[str] = []
    for variant in potrace_variants(mask):
        try:
            mask_png = mask_to_trace_bitmap(variant.mask)
            svg = normalize_svg(trace_with_potrace(mask_png, stroke_color, variant.options), stroke_color, size)
            html = inline_svg_html(svg)
            rendered = render_svg_transparent(svg, size)
            background = inpaint_background(source, variant.mask)
            composite = composite_rgba_over_rgb(rendered, background)
            metrics = score_pair(source, composite)
            mask_quality = evaluate_mask_quality(variant.mask)
            selection_score = metrics["priority_score"] + mask_quality["penalty"] + variant.penalty
            if name.startswith("nn-gated-unet"):
                selection_score -= 0.018
            overlay = Image.blend(source, diff_heatmap(source, composite), 0.38)
            candidates.append(
                TraceCandidate(
                    name=f"{name}+{variant.name}",
                    mask=variant.mask,
                    svg=svg,
                    html=html,
                    rendered=transparent_preview(rendered, size),
                    inpainted_background=background,
                    composite=composite,
                    diff_overlay=overlay,
                    metrics=metrics,
                    selection_score=selection_score,
                    mask_quality=mask_quality,
                    stroke_color=stroke_color,
                )
            )
        except Exception as exc:
            errors.append(f"{variant.name}: {exc}")

    if not candidates:
        raise RuntimeError("; ".join(errors) or "No Potrace variant succeeded")
    candidates.sort(key=lambda candidate: candidate.selection_score)
    return candidates[0]


def mask_candidates(image: Image.Image, max_masks: int = 10) -> List[Tuple[str, np.ndarray]]:
    cleaned = all_mask_candidates(image)
    cleaned.sort(key=lambda item: mask_priority(item[0], item[1]))
    return cleaned[:max_masks]


def all_mask_candidates(image: Image.Image) -> List[Tuple[str, np.ndarray]]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_rgb = np.asarray(estimate_border_color(image), dtype=np.float32)
    bg_lab = np.median(border_pixels(lab), axis=0)

    raw: List[Tuple[str, np.ndarray]] = []

    adaptive_mask = adaptive_gray_otsu_mask(image)
    raw.append(("adaptive-gray-otsu", adaptive_mask))
    raw.append(("constant-color-stroke", constant_color_stroke_mask(image)))
    raw.append(("constant-color-warm-cutoff", constant_color_warm_cutoff_mask(image)))
    raw.append(("constant-color-rmwhite-cutoff", constant_color_rmwhite_cutoff_mask(image)))
    raw.append(("constant-color-rmwhite-flood-cutoff", constant_color_rmwhite_flood_cutoff_mask(image)))
    raw.append(("constant-color-alpha-flood-purple-near-orange", constant_color_alpha_flood_purple_near_orange_mask(image)))
    raw.append(("constant-color-spectral-rmwhite-flood-cutoff", constant_color_spectral_rmwhite_flood_cutoff_mask(image)))
    raw.append(("constant-color-spectral-rmwhite-direct-cutoff", constant_color_spectral_rmwhite_direct_cutoff_mask(image)))
    nn_mask = nn_gated_unet_stroke_mask(image)
    if nn_mask is not None:
        raw.append(("nn-gated-unet", nn_mask))
    raw.append(("frangi-vesselness-stroke", frangi_vesselness_stroke_mask(image)))
    raw.append(("fraz-line-stroke", fraz_line_stroke_mask(image)))
    raw.append(("vessel-filter-stroke", vessel_filter_stroke_mask(image)))
    fused_mask = svm_fused_stroke_mask(image, adaptive_mask)
    if fused_mask is not None:
        raw.append(("adaptive-svm-fused", fused_mask))
    svm_mask = svm_pixel_stroke_mask(image)
    if svm_mask is not None:
        raw.append(("svm-pixel-stroke", svm_mask))
    raw.append(("chromatic-evidence-stroke", chromatic_evidence_stroke_mask(image)))
    add_threshold_pair(raw, "gray-otsu", gray, threshold_otsu(gray))
    try:
        sauvola = threshold_sauvola(gray, window_size=25, k=0.12)
        raw.append(("sauvola-light", gray > sauvola))
        raw.append(("sauvola-dark", gray < sauvola))
    except Exception:
        pass

    rgb_distance = np.linalg.norm(rgb.astype(np.float32) - bg_rgb[None, None, :], axis=2)
    lab_distance = np.linalg.norm(lab - bg_lab[None, None, :], axis=2)
    add_distance_masks(raw, "rgb-border", rgb_distance)
    add_distance_masks(raw, "lab-border", lab_distance)

    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    residual = gray - blur
    abs_residual = np.abs(residual)
    add_distance_masks(raw, "local-contrast", abs_residual)
    if np.percentile(residual, 95) > 2:
        raw.append(("local-light", residual > np.percentile(residual, 91)))
    if np.percentile(residual, 5) < -2:
        raw.append(("local-dark", residual < np.percentile(residual, 9)))

    expanded: List[Tuple[str, np.ndarray]] = []
    for name, candidate in raw:
        mask = clean_mask(candidate)
        expanded.append((name, mask))
        refined = refine_icon_components(mask, rgb, lab)
        if refined is not None:
            expanded.append((f"{name}-icon", refined))

    cleaned: List[Tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for name, mask in expanded:
        area = float(mask.mean())
        if area < 0.006 or area > 0.42:
            continue
        key = cv2.resize(mask.astype(np.uint8), (32, 32), interpolation=cv2.INTER_NEAREST).tobytes()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((name, mask))

    return cleaned


def fixed_mask_candidate(image: Image.Image, strategy: str) -> Tuple[str, np.ndarray]:
    if strategy == "adaptive-gray-otsu":
        return strategy, adaptive_gray_otsu_mask(image)
    if strategy == "constant-color-stroke":
        return strategy, constant_color_stroke_mask(image)
    if strategy == "constant-color-warm-cutoff":
        return strategy, constant_color_warm_cutoff_mask(image)
    if strategy == "constant-color-rmwhite-cutoff":
        return strategy, constant_color_rmwhite_cutoff_mask(image)
    if strategy == "constant-color-rmwhite-flood-cutoff":
        return strategy, constant_color_rmwhite_flood_cutoff_mask(image)
    if strategy == "constant-color-alpha-flood-purple-near-orange":
        return strategy, constant_color_alpha_flood_purple_near_orange_mask(image)
    if strategy == "constant-color-spectral-rmwhite-flood-cutoff":
        return strategy, constant_color_spectral_rmwhite_flood_cutoff_mask(image)
    if strategy == "constant-color-spectral-rmwhite-direct-cutoff":
        return strategy, constant_color_spectral_rmwhite_direct_cutoff_mask(image)
    if strategy == "frangi-vesselness-stroke":
        return strategy, frangi_vesselness_stroke_mask(image)
    if strategy == "fraz-line-stroke":
        return strategy, fraz_line_stroke_mask(image)
    if strategy == "vessel-filter-stroke":
        return strategy, vessel_filter_stroke_mask(image)
    if strategy == "adaptive-svm-fused":
        base = adaptive_gray_otsu_mask(image)
        mask = svm_fused_stroke_mask(image, base)
        if mask is None:
            return "adaptive-gray-otsu", base
        return strategy, mask
    if strategy == "svm-pixel-stroke":
        mask = svm_pixel_stroke_mask(image)
        if mask is None:
            raise KeyError("Pixel SVM model is not trained yet")
        return strategy, mask
    if strategy == "chromatic-evidence-stroke":
        return strategy, chromatic_evidence_stroke_mask(image)
    if strategy == "nn-gated-unet":
        mask = nn_gated_unet_stroke_mask(image)
        if mask is None:
            raise KeyError("NN gated U-Net model is not trained yet")
        return strategy, mask
    candidates = dict(all_mask_candidates(image))
    if strategy in candidates:
        return strategy, candidates[strategy]
    if strategy.endswith("-icon"):
        fallback = strategy.removesuffix("-icon")
        if fallback in candidates:
            return fallback, candidates[fallback]
    raise KeyError(f"Mask strategy {strategy!r} did not produce a valid candidate")


def adaptive_gray_otsu_mask(image: Image.Image) -> np.ndarray:
    """One background-normalized Otsu mask for varied backgrounds.

    This keeps the spirit of gray-otsu-dark, but applies it to local contrast
    rather than raw luminance. The primary branch is polarity: strokes can be
    darker or lighter than their immediate background.

    A conservative chromatic rescue runs after the gray mask. It handles the
    failure mode where a colorful/patterned background erases luminance contrast
    but the stroke remains separable in local RGB/Lab residual space.
    """

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background = estimate_smooth_gray_background(gray)
    residual = gray - background

    candidates = []
    for polarity, strength in [
        ("dark", np.maximum(-residual, 0)),
        ("light", np.maximum(residual, 0)),
    ]:
        high_mask = otsu_positive_residual_mask(strength)
        if high_mask is None:
            continue
        candidate_masks = [("core", high_mask)]
        expanded = hysteresis_positive_residual_mask(strength)
        if expanded is not None:
            candidate_masks.append(("hysteresis", expanded))

        for variant, mask in candidate_masks:
            mask = clean_mask(mask)
            mask = suppress_background_texture_components(mask)
            if variant == "core":
                refined = refine_icon_components(mask, rgb, lab)
                if refined is not None and should_use_refined_mask(mask, refined):
                    mask = refined
            elif not should_use_hysteresis_mask(high_mask, mask):
                continue
            area = float(mask.mean())
            if area < 0.006 or area > 0.42:
                continue
            quality = evaluate_mask_quality(mask)
            contrast = float(np.median(strength[mask > 0])) if np.any(mask) else 0.0
            score = adaptive_mask_score(quality, contrast, polarity)
            if variant == "hysteresis":
                score += 0.004
            candidates.append((score, mask, strength))

    if not candidates:
        fallback = gray < threshold_otsu(gray)
        return clean_mask(fallback)

    candidates.sort(key=lambda item: item[0])
    _, best_mask, best_strength = candidates[0]
    best_mask = recover_stroke_mask_if_confident(best_mask, best_strength, lab)
    best_mask = rescue_chromatic_mask_if_confident(best_mask, rgb, lab)
    best_mask = rescue_evidence_stroke_mask_if_confident(best_mask, rgb, lab)
    return rescue_centerline_tube_mask_if_confident(best_mask, rgb, lab)


def estimate_smooth_gray_background(gray: np.ndarray) -> np.ndarray:
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    median = cv2.medianBlur(gray_u8, 25).astype(np.float32)
    smooth = cv2.GaussianBlur(median, (0, 0), sigmaX=6.0, sigmaY=6.0)
    return smooth.astype(np.float32)


def estimate_smooth_color_background(values: np.ndarray) -> np.ndarray:
    return np.stack(
        [estimate_smooth_gray_background(values[..., channel].astype(np.float32)) for channel in range(values.shape[2])],
        axis=2,
    )


def rescue_chromatic_mask_if_confident(mask: np.ndarray, rgb: np.ndarray, lab: np.ndarray) -> np.ndarray:
    base = mask.astype(np.uint8)
    if int(base.sum()) < 20:
        return base

    candidates = []
    for name, strength in chromatic_residual_strengths(rgb, lab):
        for variant, candidate in residual_candidate_variants(strength):
            candidate = clean_mask(candidate)
            candidate = suppress_background_texture_components(candidate)
            candidates.append((name, variant, candidate))

            refined = refine_icon_components(candidate, rgb, lab)
            if refined is not None:
                candidates.append((name, f"{variant}-refined", refined.astype(np.uint8)))

    usable = []
    for name, variant, candidate in candidates:
        features = chromatic_rescue_features(base, candidate)
        if not should_use_chromatic_rescue(features):
            continue
        score = chromatic_rescue_score(features, variant)
        usable.append((score, name, variant, candidate))

    if not usable:
        return base
    usable.sort(key=lambda item: item[0])
    return usable[0][3].astype(np.uint8)


def chromatic_residual_strengths(rgb: np.ndarray, lab: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    rgb_values = rgb.astype(np.float32)
    lab_values = lab.astype(np.float32)

    lab_residual = lab_values - estimate_smooth_color_background(lab_values)
    lab_strength = np.sqrt(
        lab_residual[..., 0] ** 2
        + 1.35 * lab_residual[..., 1] ** 2
        + 1.35 * lab_residual[..., 2] ** 2
    )

    rgb_residual = np.abs(rgb_values - estimate_smooth_color_background(rgb_values))
    rgb_strength = np.max(rgb_residual, axis=2)
    return [("lab", lab_strength), ("rgb", rgb_strength)]


def residual_candidate_variants(strength: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []
    core = otsu_positive_residual_mask(strength)
    if core is not None:
        variants.append(("core", core))
    expanded = hysteresis_positive_residual_mask(strength)
    if expanded is not None:
        variants.append(("hysteresis", expanded))
    return variants


def chromatic_rescue_features(base: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    quality = evaluate_mask_quality(candidate)
    base_bool = base.astype(bool)
    candidate_bool = candidate.astype(bool)
    intersection = int(np.logical_and(base_bool, candidate_bool).sum())
    base_area = float(base.mean())
    candidate_area = float(candidate.mean())
    return {
        "area": candidate_area,
        "baseArea": base_area,
        "areaRatio": candidate_area / max(1e-6, base_area),
        "containsBase": intersection / max(1, int(base_bool.sum())),
        "baseOverlap": intersection / max(1, int(candidate_bool.sum())),
        "borderRatio": quality["borderRatio"],
        "components": quality["components"],
        "penalty": quality["penalty"],
        "strokeWidthP90": mask_stroke_width_p90(candidate),
    }


def should_use_chromatic_rescue(features: Dict[str, float]) -> bool:
    if features["area"] < 0.006 or features["area"] > 0.34:
        return False
    if features["borderRatio"] > 0.015 or features["penalty"] > 0.018:
        return False

    contained_stroke_expansion = (
        features["baseArea"] < 0.12
        and 1.80 <= features["areaRatio"] <= 2.65
        and features["containsBase"] >= 0.948
        and features["components"] <= 4
        and features["strokeWidthP90"] <= 8.2
    )
    tiny_low_contrast_seed = (
        features["baseArea"] < 0.026
        and 3.8 <= features["areaRatio"] <= 7.2
        and features["containsBase"] >= 0.55
        and features["components"] <= 4
        and 0.055 <= features["area"] <= 0.14
        and features["strokeWidthP90"] <= 5.0
    )
    return contained_stroke_expansion or tiny_low_contrast_seed


def chromatic_rescue_score(features: Dict[str, float], variant: str) -> float:
    refined_penalty = 0.001 if "refined" in variant else 0.0
    return (
        features["penalty"]
        + abs(features["areaRatio"] - 2.35) * 0.01
        + features["components"] * 0.001
        + features["strokeWidthP90"] * 0.0008
        - features["containsBase"] * 0.004
        + refined_penalty
    )


def mask_stroke_width_p90(mask: np.ndarray) -> float:
    values = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)[mask.astype(bool)]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 90))


def svm_pixel_stroke_mask(image: Image.Image) -> np.ndarray | None:
    decision = svm_pixel_decision_map(image)
    if decision is None:
        return None

    threshold = pixel_svm_threshold(decision)
    mask = decision > threshold
    mask = clean_pixel_svm_mask(mask)
    return mask.astype(np.uint8)


def svm_fused_stroke_mask(image: Image.Image, base_mask: np.ndarray | None = None) -> np.ndarray | None:
    """Use the pixel SVM as a learned stroke expansion around a precise seed.

    The adaptive mask has high precision but misses weak/colorful stroke pixels.
    The SVM has higher recall but is too eager as a standalone mask. This fused
    mask keeps the adaptive seed and adds only high-confidence SVM positives
    inside a narrow neighborhood of that seed.
    """

    decision = svm_pixel_decision_map(image)
    if decision is None:
        return None
    if base_mask is None:
        base_mask = adaptive_gray_otsu_mask(image)

    base = base_mask.astype(bool)
    positive = decision[decision > 0]
    if positive.size < 24 or int(base.sum()) < 16:
        return base_mask.astype(np.uint8)

    threshold = max(0.0, float(np.percentile(positive, 35)))
    near_seed = cv2.dilate(base.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    addition = (decision > threshold) & near_seed
    return (base | addition).astype(np.uint8)


def svm_pixel_decision_map(image: Image.Image) -> np.ndarray | None:
    model = load_pixel_svm_model()
    if model is None:
        return None

    rgb = np.asarray(image.convert("RGB"))
    h, w, _ = rgb.shape
    features = pixel_svm_features(rgb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            decision = model.decision_function(features).reshape(h, w)
        except ValueError:
            return None
    return np.nan_to_num(decision, nan=-999.0, posinf=999.0, neginf=-999.0)


def load_pixel_svm_model() -> Any | None:
    global _PIXEL_SVM_MODEL_CACHE
    if _PIXEL_SVM_MODEL_CACHE is not None:
        return _PIXEL_SVM_MODEL_CACHE
    if not PIXEL_SVM_MODEL.exists():
        return None
    import joblib

    _PIXEL_SVM_MODEL_CACHE = joblib.load(PIXEL_SVM_MODEL)
    return _PIXEL_SVM_MODEL_CACHE


def pixel_svm_features(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    rgb_u8 = rgb.astype(np.uint8)
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_unit = gray / 255.0

    background = estimate_smooth_gray_background(gray)
    gray_residual = gray - background
    dark_residual = np.maximum(-gray_residual, 0) / 255.0
    light_residual = np.maximum(gray_residual, 0) / 255.0
    abs_gray_residual = np.abs(gray_residual) / 255.0

    lab_values = lab.astype(np.float32)
    evidence = multi_scale_chromatic_evidence(rgb_u8, lab_values)
    rgb_background = estimate_smooth_color_background(rgb_u8.astype(np.float32))
    rgb_residual = np.max(np.abs(rgb_u8.astype(np.float32) - rgb_background), axis=2) / 255.0
    lab_background = estimate_smooth_color_background(lab_values)
    lab_residual = lab_values - lab_background
    lab_residual_strength = np.sqrt(
        lab_residual[..., 0] ** 2
        + 1.35 * lab_residual[..., 1] ** 2
        + 1.35 * lab_residual[..., 2] ** 2
    ) / 255.0

    gx = cv2.Sobel(gray_unit, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_unit, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    local_mean = cv2.GaussianBlur(gray_unit, (0, 0), sigmaX=2.0, sigmaY=2.0)
    local_variance = cv2.GaussianBlur((gray_unit - local_mean) ** 2, (0, 0), sigmaX=2.0, sigmaY=2.0)
    vessel_features = vessel_filter_feature_maps(gray_unit, evidence, abs_gray_residual)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx_unit = xx / max(1, w - 1)
    yy_unit = yy / max(1, h - 1)
    border_distance = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]) / max(1, min(h, w) / 2)
    adaptive_seed = adaptive_gray_otsu_mask(Image.fromarray(rgb_u8)).astype(np.float32)
    near_seed = cv2.dilate(adaptive_seed.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(np.float32)
    inside_distance = cv2.distanceTransform(adaptive_seed.astype(np.uint8), cv2.DIST_L2, 3)
    outside_distance = cv2.distanceTransform((1 - adaptive_seed).astype(np.uint8), cv2.DIST_L2, 3)
    signed_seed_distance = np.clip((inside_distance - outside_distance) / 16.0, -1.0, 1.0).astype(np.float32)

    features = [
        rgb_f[..., 0],
        rgb_f[..., 1],
        rgb_f[..., 2],
        lab[..., 0] / 255.0,
        lab[..., 1] / 255.0,
        lab[..., 2] / 255.0,
        hsv[..., 0] / 179.0,
        hsv[..., 1] / 255.0,
        hsv[..., 2] / 255.0,
        gray_unit,
        dark_residual,
        light_residual,
        abs_gray_residual,
        evidence,
        rgb_residual,
        lab_residual_strength,
        gradient,
        local_variance,
        xx_unit,
        yy_unit,
        border_distance,
        adaptive_seed,
        near_seed,
        signed_seed_distance,
        evidence * near_seed,
        abs_gray_residual * near_seed,
        rgb_residual * near_seed,
        vessel_features["frangiDark"],
        vessel_features["frangiLight"],
        vessel_features["frangiEvidence"],
        vessel_features["frangiMax"],
        vessel_features["lineEvidence"],
        vessel_features["lineResidual"],
        vessel_features["gaborEvidence"],
        vessel_features["gaborResidual"],
        vessel_features["orientationCoherence"],
        vessel_features["frangiMax"] * near_seed,
        vessel_features["lineEvidence"] * near_seed,
        vessel_features["gaborEvidence"] * near_seed,
    ]
    return np.stack(features, axis=2).reshape(-1, len(features)).astype(np.float32)


def vessel_filter_feature_maps(
    gray_unit: np.ndarray,
    evidence: np.ndarray,
    abs_gray_residual: np.ndarray,
) -> Dict[str, np.ndarray]:
    frangi_dark = frangi_response(gray_unit, black_ridges=True)
    frangi_light = frangi_response(gray_unit, black_ridges=False)
    frangi_evidence = frangi_response(evidence, black_ridges=False)
    frangi_max = np.maximum.reduce([frangi_dark, frangi_light, frangi_evidence])
    line_evidence = directional_line_strength(evidence)
    line_residual = directional_line_strength(abs_gray_residual)
    gabor_evidence = gabor_bank_response(evidence)
    gabor_residual = gabor_bank_response(abs_gray_residual)
    return {
        "frangiDark": frangi_dark,
        "frangiLight": frangi_light,
        "frangiEvidence": frangi_evidence,
        "frangiMax": frangi_max,
        "lineEvidence": line_evidence,
        "lineResidual": line_residual,
        "gaborEvidence": gabor_evidence,
        "gaborResidual": gabor_residual,
        "orientationCoherence": structure_tensor_coherence(gray_unit),
    }


def frangi_response(values: np.ndarray, black_ridges: bool) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        response = frangi(
            np.clip(values.astype(np.float32), 0.0, 1.0),
            sigmas=(0.8, 1.2, 1.8, 2.6, 3.4),
            alpha=0.5,
            beta=0.5,
            gamma=None,
            black_ridges=black_ridges,
        )
    return robust_unit_normalize(np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0))


def directional_line_strength(values: np.ndarray, lengths: Sequence[int] = (7, 11, 15)) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    local = cv2.GaussianBlur(values, (0, 0), sigmaX=3.0, sigmaY=3.0)
    best = np.zeros_like(values, dtype=np.float32)
    for length in lengths:
        radius = length // 2
        for angle in np.linspace(0, np.pi, 8, endpoint=False):
            kernel = np.zeros((length, length), dtype=np.float32)
            dx = math.cos(float(angle)) * radius
            dy = math.sin(float(angle)) * radius
            p0 = (int(round(radius - dx)), int(round(radius - dy)))
            p1 = (int(round(radius + dx)), int(round(radius + dy)))
            cv2.line(kernel, p0, p1, 1.0, 1)
            kernel /= max(1e-6, float(kernel.sum()))
            response = cv2.filter2D(values, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            best = np.maximum(best, response - local)
    return robust_unit_normalize(np.maximum(best, 0.0))


def gabor_bank_response(values: np.ndarray) -> np.ndarray:
    centered = np.clip(values.astype(np.float32), 0.0, 1.0)
    centered = centered - cv2.GaussianBlur(centered, (0, 0), sigmaX=3.0, sigmaY=3.0)
    best = np.zeros_like(centered, dtype=np.float32)
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        for sigma, wavelength in ((1.6, 4.5), (2.4, 6.5)):
            kernel = cv2.getGaborKernel((15, 15), sigma, float(theta), wavelength, 0.35, 0, ktype=cv2.CV_32F)
            kernel -= float(kernel.mean())
            denom = float(np.sum(np.abs(kernel)))
            if denom > 1e-6:
                kernel /= denom
            response = np.abs(cv2.filter2D(centered, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT))
            best = np.maximum(best, response)
    return robust_unit_normalize(best)


def structure_tensor_coherence(values: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(values.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=2.0, sigmaY=2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=2.0, sigmaY=2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=2.0, sigmaY=2.0)
    numerator = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    denominator = jxx + jyy + 1e-6
    return np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)


def pixel_svm_threshold(decision: np.ndarray) -> float:
    positive = decision[decision > 0]
    if positive.size < 24:
        return 0.0
    return max(0.0, float(np.percentile(positive, 12)))


def clean_pixel_svm_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.astype(np.uint8)
    out = cv2.medianBlur(out * 255, 3) > 0
    return out.astype(np.uint8)


def nn_gated_unet_stroke_mask(image: Image.Image) -> np.ndarray | None:
    if not NN_GATED_UNET_MODEL.exists():
        return None

    try:
        import torch
        from train_aux_fusion_icon_segmenter import TinyGatedUNet, build_feature_stack, choose_device
    except Exception:
        return None

    global _NN_GATED_UNET_CACHE
    source = canonical_image(image.convert("RGB"), SIZE)
    feature = build_feature_stack(source).astype(np.float32)
    device = choose_device()

    if _NN_GATED_UNET_CACHE is None or _NN_GATED_UNET_CACHE["device"] != device:
        checkpoint = torch.load(NN_GATED_UNET_MODEL, map_location=device)
        base = int(checkpoint.get("base", 24))
        feature_count = int(checkpoint.get("feature_count", feature.shape[0]))
        model = TinyGatedUNet(in_main=4, in_aux=feature_count - 4, base=base).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        _NN_GATED_UNET_CACHE = {
            "device": device,
            "model": model,
            "threshold": float(checkpoint.get("threshold", 0.5)),
            "feature_count": feature_count,
        }

    if feature.shape[0] != _NN_GATED_UNET_CACHE["feature_count"]:
        return None

    with torch.no_grad():
        x = torch.from_numpy(feature[None, ...]).to(device)
        logits, _ = _NN_GATED_UNET_CACHE["model"](x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = prob >= float(_NN_GATED_UNET_CACHE["threshold"])
    return clean_mask(mask.astype(np.uint8))


def chromatic_evidence_stroke_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    evidence = multi_scale_chromatic_evidence(rgb, lab)
    candidates = chromatic_evidence_candidates(evidence, rgb, lab)
    if not candidates:
        return np.zeros((image.height, image.width), dtype=np.uint8)
    candidates.sort(key=lambda item: evidence_candidate_score(item[2], item[0]))
    return candidates[0][1].astype(np.uint8)


def frangi_vesselness_stroke_mask(image: Image.Image) -> np.ndarray:
    rgb, lab, maps = image_vessel_filter_maps(image)
    return response_stroke_mask(maps["frangiMax"], rgb, lab)


def fraz_line_stroke_mask(image: Image.Image) -> np.ndarray:
    rgb, lab, maps = image_vessel_filter_maps(image)
    response = np.maximum.reduce([maps["lineEvidence"], maps["lineResidual"], maps["gaborEvidence"], maps["gaborResidual"]])
    return response_stroke_mask(response, rgb, lab)


def vessel_filter_stroke_mask(image: Image.Image) -> np.ndarray:
    rgb, lab, maps = image_vessel_filter_maps(image)
    response = np.maximum.reduce(
        [
            maps["frangiMax"],
            maps["lineEvidence"],
            maps["lineResidual"],
            maps["gaborEvidence"],
            maps["gaborResidual"],
        ]
    )
    return response_stroke_mask(response, rgb, lab)


def image_vessel_filter_maps(image: Image.Image) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rgb = np.asarray(image.convert("RGB"))
    rgb_u8 = rgb.astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_unit = gray / 255.0
    background = estimate_smooth_gray_background(gray)
    abs_gray_residual = np.abs(gray - background) / 255.0
    evidence = multi_scale_chromatic_evidence(rgb_u8, lab)
    return rgb, lab, vessel_filter_feature_maps(gray_unit, evidence, abs_gray_residual)


def response_stroke_mask(response: np.ndarray, rgb: np.ndarray, lab: np.ndarray) -> np.ndarray:
    response = np.clip(response.astype(np.float32), 0.0, 1.0)
    candidates = chromatic_evidence_candidates(response, rgb, lab)
    if candidates:
        candidates.sort(key=lambda item: evidence_candidate_score(item[2], item[0]))
        return candidates[0][1].astype(np.uint8)

    positive = response[response > 0.02]
    if positive.size < 24:
        return np.zeros(response.shape, dtype=np.uint8)
    threshold = max(float(np.percentile(positive, 82)), float(threshold_otsu(positive)))
    return suppress_background_texture_components(clean_mask(response >= threshold)).astype(np.uint8)


def constant_color_stroke_mask(image: Image.Image) -> np.ndarray:
    alpha, _ = constant_color_alpha_map(image)
    positive = alpha[alpha > 0.02]
    if positive.size < 24:
        return np.zeros(alpha.shape, dtype=np.uint8)
    threshold = max(0.18, float(np.percentile(positive, 72)))
    return suppress_background_texture_components(clean_mask(alpha >= threshold)).astype(np.uint8)


def constant_color_warm_cutoff_mask(image: Image.Image) -> np.ndarray:
    alpha, _ = constant_color_alpha_map(image)
    return warm_high_intensity_alpha_mask(alpha)


def constant_color_rmwhite_cutoff_mask(image: Image.Image) -> np.ndarray:
    alpha, _ = constant_color_alpha_map(image)
    return rmwhite_style_alpha_cutoff_mask(alpha)


def constant_color_rmwhite_flood_cutoff_mask(image: Image.Image) -> np.ndarray:
    alpha, _ = constant_color_alpha_map(image)
    return rmwhite_flood_alpha_cutoff_mask(alpha)


def constant_color_alpha_flood_purple_near_orange_mask(image: Image.Image) -> np.ndarray:
    alpha, _ = constant_color_alpha_map(image)
    base = alpha_flood_without_purple_mask(alpha)
    purple = purple_near_orange_alpha_mask(alpha, base)
    mask = (base.astype(bool) | purple.astype(bool)).astype(np.uint8)
    mask = fill_tiny_white_slits(mask, max_area=24, max_span=9)
    return suppress_background_texture_components(mask).astype(np.uint8)


def constant_color_spectral_rmwhite_flood_cutoff_mask(image: Image.Image) -> np.ndarray:
    alpha, _ = constant_color_alpha_map(image)
    high_high, _, _, _ = spectral_high_high_alpha_projection(image, alpha)
    return rmwhite_flood_alpha_cutoff_mask(high_high)


def constant_color_spectral_rmwhite_direct_cutoff_mask(image: Image.Image) -> np.ndarray:
    alpha, _ = constant_color_alpha_map(image)
    high_high, _, _, _ = spectral_high_high_alpha_projection(image, alpha)
    return rmwhite_direct_bw_cutoff_mask(high_high, seed_alpha=alpha)


def warm_high_intensity_alpha_mask(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    positive = alpha[alpha > 0.02]
    if positive.size < 24:
        return np.zeros(alpha.shape, dtype=np.uint8)

    heatmap = cv2.applyColorMap((alpha * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    warm_band = ((hue <= 35) | ((saturation <= 90) & (value >= 205))) & (value >= 115)
    cutoff = max(0.18, min(0.55, float(np.percentile(positive, 45))))
    mask = (alpha >= cutoff) & warm_band
    return suppress_background_texture_components(clean_mask(mask)).astype(np.uint8)


def rmwhite_style_alpha_cutoff_mask(alpha: np.ndarray, fuzz: int = 18) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    positive = alpha[alpha > 0.02]
    if positive.size < 24:
        return np.zeros(alpha.shape, dtype=np.uint8)

    score = rmwhite_matte_alpha_from_projection(alpha, fuzz=fuzz)
    warm = warm_high_intensity_alpha_band(alpha)
    seed = (score >= 0.46) & warm
    if int(seed.sum()) < 16:
        seed = score >= 0.46

    support = score >= 0.11
    count, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    if count <= 1:
        mask = support
    else:
        seed_labels = np.unique(labels[seed])
        seed_labels = seed_labels[seed_labels > 0]
        mask = np.isin(labels, seed_labels) if seed_labels.size else np.zeros(alpha.shape, dtype=bool)

    mask = clean_mask(mask)
    mask = fill_tiny_white_slits(mask, max_area=24, max_span=9)
    return suppress_background_texture_components(mask).astype(np.uint8)


def rmwhite_flood_alpha_cutoff_mask(alpha: np.ndarray, fuzz: int = 18) -> np.ndarray:
    matte = rmwhite_flood_filtered_alpha(alpha, fuzz=fuzz)
    positive = matte[matte > 0.02]
    if positive.size < 24:
        return np.zeros(matte.shape, dtype=np.uint8)

    warm = warm_high_intensity_alpha_band(alpha)
    seed = (matte >= max(0.38, float(np.percentile(positive, 72)))) & warm
    if int(seed.sum()) < 16:
        seed = matte >= max(0.38, float(np.percentile(positive, 78)))
    support = matte >= 0.08
    mask = seeded_connected_components(support, seed)
    mask = clean_mask(mask)
    mask = fill_tiny_white_slits(mask, max_area=24, max_span=9)
    return suppress_background_texture_components(mask).astype(np.uint8)


def alpha_flood_without_purple_mask(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    positive = alpha[alpha > 0.02]
    if positive.size < 24:
        return np.zeros(alpha.shape, dtype=np.uint8)

    purple = purple_alpha_band(alpha)
    seed = orange_yellow_alpha_anchor_band(alpha)
    if int(seed.sum()) < 12:
        warm = warm_high_intensity_alpha_band(alpha)
        warm_values = alpha[warm]
        if warm_values.size >= 16:
            seed_threshold = max(0.38, float(np.percentile(warm_values, 40)))
        else:
            seed_threshold = max(0.45, float(np.percentile(positive, 82)))
        seed = warm & (alpha >= seed_threshold)

    support_threshold = max(0.045, float(np.percentile(positive, 12)))
    support = (alpha >= support_threshold) & ~purple
    mask = seeded_connected_components(support, seed)
    mask = fill_tiny_white_slits(mask, max_area=24, max_span=9)
    return suppress_background_texture_components(mask).astype(np.uint8)


def purple_near_orange_alpha_mask(alpha: np.ndarray, base: np.ndarray, radius: int = 7) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    positive = alpha[alpha > 0.02]
    if positive.size < 24:
        return np.zeros(alpha.shape, dtype=np.uint8)

    purple = purple_alpha_band(alpha)
    anchor = orange_yellow_alpha_anchor_band(alpha)
    search_seed = (anchor | base.astype(bool)).astype(np.uint8)
    if int(search_seed.sum()) == 0:
        return np.zeros(alpha.shape, dtype=np.uint8)

    distance = cv2.distanceTransform((1 - search_seed).astype(np.uint8), cv2.DIST_L2, 3)
    minimum_alpha = max(0.025, float(np.percentile(positive, 8)))
    return (purple & (distance <= radius) & (alpha >= minimum_alpha)).astype(np.uint8)


def orange_yellow_alpha_anchor_band(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    warm = warm_high_intensity_alpha_band(alpha)
    purple = purple_alpha_band(alpha)
    candidate = warm & ~purple
    positive = alpha[alpha > 0.02]
    if positive.size < 24:
        return np.zeros(alpha.shape, dtype=bool)

    values = alpha[candidate]
    if values.size >= 16:
        threshold = max(0.32, float(np.percentile(values, 22)))
    else:
        threshold = max(0.38, float(np.percentile(positive, 72)))
    return candidate & (alpha >= threshold)


def rmwhite_flood_filtered_alpha(alpha: np.ndarray, fuzz: int = 18) -> np.ndarray:
    cleaned_projection = purple_removed_alpha_projection(alpha)
    score = rmwhite_matte_alpha_from_projection(cleaned_projection, fuzz=fuzz)
    positive = score[score > 0.02]
    if positive.size < 24:
        return np.zeros(cleaned_projection.shape, dtype=np.float32)
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def rmwhite_direct_bw_cutoff_mask(
    evidence: np.ndarray,
    fuzz: int = 18,
    seed_alpha: np.ndarray | None = None,
) -> np.ndarray:
    return hysteresis_gray_flood_mask(evidence, fuzz=fuzz, seed_alpha=seed_alpha)


def hysteresis_gray_flood_mask(
    evidence: np.ndarray,
    fuzz: int = 18,
    seed_alpha: np.ndarray | None = None,
) -> np.ndarray:
    matte = rmwhite_matte_alpha_from_projection(evidence, fuzz=fuzz)
    positive = matte[matte > 0.02]
    if positive.size < 24:
        return np.zeros(matte.shape, dtype=np.uint8)

    seed_threshold = max(0.26, float(np.percentile(positive, 42)))
    support_threshold = max(0.055, min(seed_threshold * 0.36, float(np.percentile(positive, 14))))
    seed = matte >= seed_threshold
    if seed_alpha is not None:
        warm_seed = warm_high_intensity_alpha_band(seed_alpha)
        gated_seed = seed & warm_seed
        if int(gated_seed.sum()) >= 12:
            seed = gated_seed
    support = matte >= support_threshold
    mask = seeded_connected_components(support, seed)
    mask = clean_mask(mask)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    mask = fill_tiny_white_slits(mask, max_area=24, max_span=9)
    return suppress_background_texture_components(mask).astype(np.uint8)


def spectral_high_high_alpha_projection(
    image: Image.Image,
    alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    candidates = spectral_channel_candidates(image, alpha)
    if not candidates:
        return alpha, alpha, "raw-alpha", 0.0
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, name, channel = candidates[0]
    return np.minimum(alpha, channel).astype(np.float32), channel.astype(np.float32), name, float(score)


def spectral_channel_candidates(image: Image.Image, alpha: np.ndarray) -> List[Tuple[float, str, np.ndarray]]:
    rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_background = estimate_smooth_color_background_scaled(lab, 31, 8.0)
    lab_residual = lab - lab_background
    ab = lab_residual[..., 1:3]

    candidates: List[Tuple[float, str, np.ndarray]] = []
    for index, angle in enumerate(np.linspace(0, 2 * math.pi, 24, endpoint=False)):
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        response = np.maximum(0.0, ab[..., 0] * direction[0] + ab[..., 1] * direction[1])
        channel = robust_unit_normalize(response)
        candidates.append((spectral_channel_score(channel, alpha), f"lab{index:02d}", channel))

    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_background = estimate_smooth_gray_background(gray)
    dark = robust_unit_normalize(np.maximum(gray_background - gray, 0.0))
    light = robust_unit_normalize(np.maximum(gray - gray_background, 0.0))
    candidates.append((spectral_channel_score(dark, alpha), "gray-dark", dark))
    candidates.append((spectral_channel_score(light, alpha), "gray-light", light))
    return candidates


def spectral_channel_score(channel: np.ndarray, alpha: np.ndarray) -> float:
    channel = np.clip(channel.astype(np.float32), 0.0, 1.0)
    warm = warm_high_intensity_alpha_band(alpha)
    purple = purple_alpha_band(alpha)
    warm_weight = np.where(warm, alpha**1.35, 0.0)
    purple_weight = np.where(purple, alpha**1.35, 0.0)
    warm_fit = signed_weighted_mean(channel, warm_weight)
    purple_fit = signed_weighted_mean(channel, purple_weight)
    signed_fit = signed_weighted_mean(channel, warm_weight - purple_weight)
    energy = float(channel.sum()) + 1e-6
    warm_ratio = float((channel * warm.astype(np.float32)).sum() / energy)
    purple_ratio = float((channel * purple.astype(np.float32)).sum() / energy)
    border_penalty = spectral_border_texture_penalty(threshold_spectral_channel(channel))
    area = float(threshold_spectral_channel(channel).mean())
    area_penalty = abs(area - 0.12)
    return float(
        signed_fit * 2.2
        + warm_fit * 1.4
        + warm_ratio * 0.75
        - purple_fit * 1.7
        - purple_ratio * 1.6
        - border_penalty * 1.1
        - area_penalty * 0.35
    )


def signed_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.abs(weights).sum())
    if denom <= 1e-6:
        return 0.0
    return float((values * weights).sum() / denom)


def threshold_spectral_channel(channel: np.ndarray) -> np.ndarray:
    positive = channel[channel > 0.02]
    if positive.size < 24:
        return np.zeros(channel.shape, dtype=np.uint8)
    try:
        threshold = float(threshold_otsu(positive))
    except Exception:
        threshold = float(np.percentile(positive, 72))
    return (channel >= max(0.18, min(0.82, threshold))).astype(np.uint8)


def spectral_border_texture_penalty(mask: np.ndarray) -> float:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    total = int(mask.sum())
    if total == 0:
        return 1.0
    h, w = mask.shape
    bad_area = 0
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches = x <= 1 or y <= 1 or x + ww >= w - 2 or y + hh >= h - 2
        spans = ww >= w * 0.62 or hh >= h * 0.62
        if touches and spans:
            bad_area += area
    return float(bad_area / max(1, total))


def purple_removed_alpha_projection(alpha: np.ndarray) -> np.ndarray:
    alpha = warm_normalized_alpha_projection(alpha)
    positive = alpha[alpha > 0.02]
    if positive.size < 24:
        return np.zeros(alpha.shape, dtype=np.float32)

    warm = warm_high_intensity_alpha_band(alpha)
    purple = purple_alpha_band(alpha)
    seed_threshold = max(0.38, float(np.percentile(positive, 78)))
    seed = (alpha >= seed_threshold) & warm
    if int(seed.sum()) < 16:
        seed = alpha >= max(0.38, float(np.percentile(positive, 84)))

    support = (alpha >= 0.045) & ~purple
    core = seeded_connected_components(support, seed).astype(np.uint8)
    if int(core.sum()) == 0:
        return np.zeros(alpha.shape, dtype=np.float32)

    halo = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1).astype(bool)
    cleaned = alpha.copy()
    cleaned[~halo] = 0.0
    cleaned[purple] = 0.0
    return np.clip(cleaned, 0.0, 1.0).astype(np.float32)


def warm_normalized_alpha_projection(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    positive = alpha[alpha > 0.02]
    if positive.size < 24:
        return np.zeros(alpha.shape, dtype=np.float32)

    warm = warm_high_intensity_alpha_band(alpha)
    if int(warm.sum()) < 16:
        return alpha

    purple = purple_alpha_band(alpha)
    warm_values = alpha[warm]
    seed_threshold = max(0.58, float(np.percentile(warm_values, 35)))
    seed = warm & (alpha >= seed_threshold)
    support = (alpha >= 0.045) & ~purple
    keep = seeded_connected_components(support, seed).astype(bool)
    if int(keep.sum()) < 16:
        keep = remove_small_objects((warm & ~purple).astype(bool), min_size=4)

    normalized = alpha.copy()
    normalized[keep] = 210.0 / 255.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def seeded_connected_components(support: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    if count <= 1:
        return support.astype(np.uint8)
    seed_labels = np.unique(labels[seed.astype(bool)])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return np.zeros(support.shape, dtype=np.uint8)
    return np.isin(labels, seed_labels).astype(np.uint8)


def rmwhite_matte_alpha_from_projection(alpha: np.ndarray, fuzz: int = 18) -> np.ndarray:
    white_matte_value = 255.0 * (1.0 - np.clip(alpha.astype(np.float32), 0.0, 1.0))
    white_matte_value = np.minimum(white_matte_value + float(fuzz), 255.0)
    return np.clip((255.0 - white_matte_value) / max(1.0, 255.0 - float(fuzz)), 0.0, 1.0).astype(np.float32)


def warm_high_intensity_alpha_band(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    heatmap = cv2.applyColorMap((alpha * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    return ((((hue <= 45) | (hue >= 173)) | ((saturation <= 110) & (value >= 198))) & (value >= 72)).astype(bool)


def purple_alpha_band(alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    heatmap = cv2.applyColorMap((alpha * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    hsv = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    return ((hue >= 108) & (hue <= 179) & (saturation >= 45) & (value >= 4)).astype(bool)


def fill_tiny_white_slits(mask: np.ndarray, max_area: int, max_span: int) -> np.ndarray:
    foreground = mask.astype(bool)
    background = ~foreground
    h, w = foreground.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background.astype(np.uint8), connectivity=8)
    out = foreground.copy()
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches_border and area <= max_area and max(ww, hh) <= max_span:
            out[labels == label] = True
    return out.astype(np.uint8)


def constant_color_alpha_map(image: Image.Image) -> Tuple[np.ndarray, str]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    evidence = multi_scale_chromatic_evidence(rgb_u8, lab)
    background = estimate_smooth_color_background_scaled(rgb, 31, 8.0)
    stroke = estimate_constant_stroke_rgb(rgb, evidence)

    stroke_field = stroke[None, None, :].astype(np.float32)
    direction = stroke_field - background
    observed = rgb - background
    denom = np.sum(direction * direction, axis=2)
    alpha = np.sum(observed * direction, axis=2) / np.maximum(denom, 36.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    reconstructed = background + alpha[..., None] * direction
    error = np.linalg.norm(rgb - reconstructed, axis=2) / 255.0
    confidence = np.exp(-np.square(error / 0.12))
    alpha = np.clip(alpha * confidence, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), sigmaX=0.45, sigmaY=0.45)
    return alpha.astype(np.float32), rgb_hex(stroke)


def estimate_constant_stroke_rgb(rgb: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    seed = adaptive_gray_otsu_mask(image).astype(bool)
    if int(seed.sum()) >= 24:
        seed_values = evidence[seed]
        if seed_values.size:
            seed = seed & (evidence >= max(0.05, float(np.percentile(seed_values, 35))))

    if int(seed.sum()) < 24:
        positive = evidence[evidence > 0.02]
        if positive.size:
            seed = evidence >= float(np.percentile(positive, 88))

    if int(seed.sum()) < 24:
        return np.median(rgb.reshape(-1, 3), axis=0)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    h, w = seed.shape
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    best_label = 0
    best_score = -1.0
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        centroid = np.asarray(centroids[label], dtype=np.float32)
        center_score = 1.0 - min(0.9, float(np.linalg.norm((centroid - center) / max(w, h))))
        evidence_score = float(np.median(evidence[labels == label]))
        score = area * (0.4 + center_score) * (0.25 + evidence_score)
        if score > best_score:
            best_score = score
            best_label = label

    selected = labels == best_label if best_label else seed
    pixels = rgb[selected]
    if pixels.size == 0:
        pixels = rgb[seed]
    return np.median(pixels, axis=0)


def rescue_evidence_stroke_mask_if_confident(mask: np.ndarray, rgb: np.ndarray, lab: np.ndarray) -> np.ndarray:
    base = mask.astype(np.uint8)
    evidence = multi_scale_chromatic_evidence(rgb, lab)
    usable = []
    for name, candidate, features in chromatic_evidence_candidates(evidence, rgb, lab):
        relation = mask_relation_features(base, candidate)
        if not should_use_evidence_rescue(relation, features):
            continue
        score = evidence_candidate_score(features, name)
        usable.append((score, name, candidate))

    if not usable:
        return base
    usable.sort(key=lambda item: item[0])
    return usable[0][2].astype(np.uint8)


def multi_scale_chromatic_evidence(rgb: np.ndarray, lab: np.ndarray) -> np.ndarray:
    rgb_values = rgb.astype(np.float32)
    lab_values = lab.astype(np.float32)
    scales = [(11, 2.5), (23, 5.5), (41, 9.0)]
    responses = []
    for kernel_size, sigma in scales:
        lab_residual = lab_values - estimate_smooth_color_background_scaled(lab_values, kernel_size, sigma)
        rgb_residual = np.abs(rgb_values - estimate_smooth_color_background_scaled(rgb_values, kernel_size, sigma))
        lab_strength = np.sqrt(
            lab_residual[..., 0] ** 2
            + 1.45 * lab_residual[..., 1] ** 2
            + 1.45 * lab_residual[..., 2] ** 2
        )
        rgb_strength = np.max(rgb_residual, axis=2)
        responses.append(np.maximum(robust_unit_normalize(lab_strength), robust_unit_normalize(rgb_strength)))
    evidence = np.max(np.stack(responses, axis=0), axis=0)
    evidence = cv2.GaussianBlur(evidence.astype(np.float32), (0, 0), sigmaX=0.65, sigmaY=0.65)
    return np.clip(evidence, 0.0, 1.0).astype(np.float32)


def estimate_smooth_color_background_scaled(values: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    channels = []
    for channel in range(values.shape[2]):
        channel_values = np.clip(values[..., channel], 0, 255).astype(np.uint8)
        median = cv2.medianBlur(channel_values, kernel_size).astype(np.float32)
        smooth = cv2.GaussianBlur(median, (0, 0), sigmaX=sigma, sigmaY=sigma)
        channels.append(smooth)
    return np.stack(channels, axis=2).astype(np.float32)


def robust_unit_normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values.astype(np.float32), [50, 99])
    return np.clip((values - low) / max(1e-6, high - low), 0.0, 1.0).astype(np.float32)


def chromatic_evidence_candidates(
    evidence: np.ndarray,
    rgb: np.ndarray,
    lab: np.ndarray,
) -> List[Tuple[str, np.ndarray, Dict[str, float]]]:
    positive = evidence[evidence > 0.02]
    if positive.size < 24:
        return []

    candidates: List[Tuple[str, np.ndarray, Dict[str, float]]] = []
    low_floor = float(np.percentile(positive, 45))
    thresholds = sorted({float(np.percentile(positive, percentile)) for percentile in (82, 88, 92, 95)})
    for high in thresholds:
        for low_factor in (0.32, 0.42, 0.55, 0.68):
            raw = evidence_hysteresis_mask(evidence, high, max(low_floor, high * low_factor))
            if raw is None:
                continue
            add_evidence_candidate(candidates, f"hysteresis-{high:.2f}-{low_factor}", raw, evidence, rgb, lab)

    grabcut = evidence_grabcut_mask(evidence)
    if grabcut is not None:
        add_evidence_candidate(candidates, "grabcut", grabcut, evidence, rgb, lab)
    return candidates


def add_evidence_candidate(
    candidates: List[Tuple[str, np.ndarray, Dict[str, float]]],
    name: str,
    mask: np.ndarray,
    evidence: np.ndarray,
    rgb: np.ndarray,
    lab: np.ndarray,
) -> None:
    cleaned = suppress_background_texture_components(clean_mask(mask))
    if 0.006 <= float(cleaned.mean()) <= 0.42:
        candidates.append((name, cleaned.astype(np.uint8), evidence_mask_features(cleaned, evidence)))

    refined = refine_icon_components(cleaned, rgb, lab)
    if refined is not None and 0.006 <= float(refined.mean()) <= 0.42:
        refined = refined.astype(np.uint8)
        candidates.append((f"{name}-refined", refined, evidence_mask_features(refined, evidence)))


def evidence_hysteresis_mask(evidence: np.ndarray, high: float, low: float) -> np.ndarray | None:
    core = evidence >= high
    support = evidence >= low
    count, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    seed_labels = np.unique(labels[core])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return None
    return np.isin(labels, seed_labels).astype(np.uint8)


def evidence_grabcut_mask(evidence: np.ndarray) -> np.ndarray | None:
    positive = evidence[evidence > 0.02]
    if positive.size < 24:
        return None

    h, w = evidence.shape
    evidence_u8 = np.clip(evidence * 255, 0, 255).astype(np.uint8)
    pseudo_image = cv2.merge(
        [
            evidence_u8,
            cv2.GaussianBlur(evidence_u8, (0, 0), sigmaX=1.5, sigmaY=1.5),
            cv2.equalizeHist(evidence_u8),
        ]
    )

    low, support, high = np.percentile(positive, [45, 72, 92])
    markers = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    border = np.zeros((h, w), dtype=bool)
    border[:3, :] = True
    border[-3:, :] = True
    border[:, :3] = True
    border[:, -3:] = True
    markers[(evidence <= low) | border] = cv2.GC_BGD
    markers[evidence >= support] = cv2.GC_PR_FGD
    markers[evidence >= high] = cv2.GC_FGD

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            pseudo_image,
            markers,
            None,
            background_model,
            foreground_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except Exception:
        return None
    return np.where((markers == cv2.GC_FGD) | (markers == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def evidence_mask_features(mask: np.ndarray, evidence: np.ndarray) -> Dict[str, float]:
    features = evaluate_mask_quality(mask)
    mask_bool = mask.astype(bool)
    inside = evidence[mask_bool]
    outside = evidence[~mask_bool]
    inside_median = float(np.median(inside)) if inside.size else 0.0
    outside_p95 = float(np.percentile(outside, 95)) if outside.size else 1.0
    features.update(
        {
            "strokeWidthP90": mask_stroke_width_p90(mask),
            "insideEvidence": inside_median,
            "evidenceMargin": inside_median / max(1e-6, outside_p95),
        }
    )
    return features


def mask_relation_features(base: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    base_bool = base.astype(bool)
    candidate_bool = candidate.astype(bool)
    intersection = int(np.logical_and(base_bool, candidate_bool).sum())
    base_area = float(base.mean())
    candidate_area = float(candidate.mean())
    return {
        "baseArea": base_area,
        "area": candidate_area,
        "areaRatio": candidate_area / max(1e-6, base_area),
        "containsBase": intersection / max(1, int(base_bool.sum())),
        "baseOverlap": intersection / max(1, int(candidate_bool.sum())),
    }


def should_use_evidence_rescue(relation: Dict[str, float], features: Dict[str, float]) -> bool:
    if features["borderRatio"] > 0.012:
        return False
    if features["components"] > 6 or features["area"] < 0.045 or features["area"] > 0.22:
        return False
    if features["strokeWidthP90"] > 8.2:
        return False

    tiny_seed_expansion = (
        relation["baseArea"] < 0.035
        and 0.05 <= features["area"] <= 0.14
        and 2.5 <= relation["areaRatio"] <= 8.0
        and relation["containsBase"] >= 0.75
        and features["insideEvidence"] >= 0.62
    )
    striped_partial_recovery = (
        relation["baseArea"] < 0.085
        and 0.075 <= features["area"] <= 0.15
        and relation["containsBase"] <= 0.50
        and relation["baseOverlap"] <= 0.30
        and features["insideEvidence"] >= 0.74
        and features["evidenceMargin"] >= 1.2
        and features["components"] <= 6
    )
    divergent_low_confidence_recovery = (
        0.065 <= relation["baseArea"] <= 0.09
        and 1.6 <= relation["areaRatio"] <= 2.1
        and 0.45 <= relation["containsBase"] <= 0.60
        and 0.24 <= relation["baseOverlap"] <= 0.36
        and features["insideEvidence"] >= 0.50
        and features["evidenceMargin"] <= 1.0
        and features["components"] <= 2
    )
    contained_high_evidence_expansion = (
        relation["baseArea"] < 0.10
        and 1.20 <= relation["areaRatio"] <= 1.80
        and relation["containsBase"] >= 0.99
        and relation["baseOverlap"] >= 0.55
        and features["evidenceMargin"] >= 3.5
        and features["insideEvidence"] >= 0.80
        and features["area"] <= 0.16
    )
    large_similar_stroke_completion = (
        0.12 <= relation["baseArea"] <= 0.18
        and 1.0 <= relation["areaRatio"] <= 1.16
        and relation["containsBase"] >= 0.94
        and relation["baseOverlap"] >= 0.80
        and features["insideEvidence"] >= 0.88
        and features["area"] <= 0.20
    )
    return (
        tiny_seed_expansion
        or striped_partial_recovery
        or divergent_low_confidence_recovery
        or contained_high_evidence_expansion
        or large_similar_stroke_completion
    )


def evidence_candidate_score(features: Dict[str, float], name: str) -> float:
    return (
        features["penalty"]
        + abs(features["area"] - 0.105) * 0.12
        + max(0.0, features["borderRatio"] - 0.005) * 0.9
        + max(0.0, features["components"] - 6) * 0.004
        + max(0.0, features["strokeWidthP90"] - 8.0) * 0.01
        - min(0.055, features["insideEvidence"] * 0.035)
        - min(0.025, features["evidenceMargin"] * 0.006)
        + (0.003 if "grabcut" in name else 0.0)
    )


def rescue_centerline_tube_mask_if_confident(mask: np.ndarray, rgb: np.ndarray, lab: np.ndarray) -> np.ndarray:
    base = mask.astype(np.uint8)
    evidence = multi_scale_chromatic_evidence(rgb, lab)
    usable = []
    for support_name, support_mask, _ in chromatic_evidence_candidates(evidence, rgb, lab):
        for radius in centerline_radius_candidates(support_mask):
            tube, centerline = centerline_tube_from_support(support_mask, radius)
            if tube is None:
                continue
            features = centerline_tube_features(base, tube, centerline, evidence)
            if not should_use_centerline_tube_rescue(features):
                continue
            score = centerline_tube_score(features, support_name)
            usable.append((score, support_name, tube))

    if not usable:
        return base
    usable.sort(key=lambda item: item[0])
    return usable[0][2].astype(np.uint8)


def centerline_radius_candidates(mask: np.ndarray) -> List[float]:
    centerline = pruned_skeleton(mask)
    if int(centerline.sum()) < 8:
        return []
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    values = distances[centerline.astype(bool)]
    if values.size < 8:
        return []

    raw = [float(np.percentile(values, percentile)) for percentile in (45, 60)]
    raw.extend([4.6, 5.2])
    radii: List[float] = []
    for value in raw:
        radius = float(np.clip(value, 1.8, 5.2))
        if all(abs(radius - existing) > 0.25 for existing in radii):
            radii.append(radius)
    return radii


def centerline_tube_from_support(mask: np.ndarray, radius: float) -> Tuple[np.ndarray | None, np.ndarray]:
    centerline = pruned_skeleton(mask)
    if int(centerline.sum()) < 8:
        return None, centerline

    radius_int = int(max(1, round(radius)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_int * 2 + 1, radius_int * 2 + 1))
    tube = cv2.dilate(centerline.astype(np.uint8), kernel, iterations=1)
    tube = clean_mask(tube)
    tube = suppress_background_texture_components(tube)
    return tube.astype(np.uint8), centerline.astype(np.uint8)


def pruned_skeleton(mask: np.ndarray) -> np.ndarray:
    centerline = skeletonize(mask.astype(bool))
    centerline = remove_small_objects(centerline, min_size=8)
    return centerline.astype(np.uint8)


def centerline_tube_features(
    base: np.ndarray,
    tube: np.ndarray,
    centerline: np.ndarray,
    evidence: np.ndarray,
) -> Dict[str, float]:
    features = evaluate_mask_quality(tube)
    base_bool = base.astype(bool)
    tube_bool = tube.astype(bool)
    centerline_bool = centerline.astype(bool)
    intersection = int(np.logical_and(base_bool, tube_bool).sum())
    inside = evidence[tube_bool]
    centerline_values = evidence[centerline_bool]
    outside = evidence[~tube_bool]
    inside_median = float(np.median(inside)) if inside.size else 0.0
    outside_p95 = float(np.percentile(outside, 95)) if outside.size else 1.0
    base_area = float(base.mean())
    tube_area = float(tube.mean())
    features.update(
        {
            "strokeWidthP90": mask_stroke_width_p90(tube),
            "insideEvidence": inside_median,
            "centerlineEvidence": float(np.median(centerline_values)) if centerline_values.size else 0.0,
            "evidenceMargin": inside_median / max(1e-6, outside_p95),
            "baseArea": base_area,
            "areaRatio": tube_area / max(1e-6, base_area),
            "containsBase": intersection / max(1, int(base_bool.sum())),
            "baseOverlap": intersection / max(1, int(tube_bool.sum())),
        }
    )
    return features


def should_use_centerline_tube_rescue(features: Dict[str, float]) -> bool:
    return (
        0.075 <= features["baseArea"] <= 0.105
        and 0.050 <= features["area"] <= 0.095
        and 0.55 <= features["areaRatio"] <= 1.10
        and features["containsBase"] <= 0.38
        and features["baseOverlap"] <= 0.60
        and features["components"] <= 3
        and features["borderRatio"] <= 0.012
        and 4.4 <= features["strokeWidthP90"] <= 6.2
        and features["insideEvidence"] >= 0.55
        and features["centerlineEvidence"] >= 0.84
        and features["evidenceMargin"] <= 0.90
    )


def centerline_tube_score(features: Dict[str, float], support_name: str) -> float:
    return (
        features["penalty"]
        + abs(features["area"] - 0.075) * 0.12
        + features["components"] * 0.002
        - min(0.03, features["centerlineEvidence"] * 0.02)
        + (0.002 if "grabcut" in support_name else 0.0)
    )


def otsu_positive_residual_mask(strength: np.ndarray) -> np.ndarray | None:
    positive = strength[strength > 1.5]
    if positive.size < 24:
        return None
    try:
        threshold = threshold_otsu(positive)
    except Exception:
        threshold = float(np.percentile(positive, 72))
    threshold = max(float(threshold), float(np.percentile(positive, 58)))
    return strength > threshold


def hysteresis_positive_residual_mask(strength: np.ndarray) -> np.ndarray | None:
    positive = strength[strength > 1.5]
    if positive.size < 24:
        return None
    try:
        high = float(threshold_otsu(positive))
    except Exception:
        high = float(np.percentile(positive, 72))
    high = max(high, float(np.percentile(positive, 58)))
    low = max(high * 0.45, float(np.percentile(positive, 32)))
    high_mask = strength > high
    low_mask = strength > low
    count, labels = cv2.connectedComponents(low_mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    seed_labels = np.unique(labels[high_mask])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return None
    return np.isin(labels, seed_labels)


def suppress_background_texture_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(np.uint8)

    h, w = mask.shape
    background_labels = grouped_band_texture_labels(stats, h, w)
    keep_labels = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_edge = x <= 1 or y <= 1 or x + ww >= w - 2 or y + hh >= h - 2
        spans_width = ww >= w * 0.82 and hh <= h * 0.20
        spans_height = hh >= h * 0.82 and ww <= w * 0.20
        huge_border_region = touches_edge and area >= h * w * 0.12
        wide_border_band = touches_edge and ww >= w * 0.25 and hh <= h * 0.16 and ww / max(1, hh) >= 3.2
        if label in background_labels:
            continue
        if touches_edge and (spans_width or spans_height or huge_border_region or wide_border_band):
            continue
        keep_labels.append(label)

    if not keep_labels:
        return np.zeros_like(mask, dtype=np.uint8)
    return np.isin(labels, keep_labels).astype(np.uint8)


def grouped_band_texture_labels(stats: np.ndarray, h: int, w: int) -> set[int]:
    bands: List[Dict[str, Any]] = []
    for label in range(1, stats.shape[0]):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        if hh > h * 0.18 or ww < w * 0.12 or ww / max(1, hh) < 2.6:
            continue
        cy = y + hh / 2
        for band in bands:
            if abs(band["cy"] - cy) <= h * 0.07:
                band["labels"].append(label)
                band["intervals"].append((x, x + ww))
                band["cy"] = (band["cy"] + cy) / 2
                break
        else:
            bands.append({"cy": cy, "labels": [label], "intervals": [(x, x + ww)]})

    background_labels: set[int] = set()
    for band in bands:
        if len(band["labels"]) < 2:
            continue
        coverage = interval_coverage(band["intervals"])
        if coverage >= w * 0.58:
            background_labels.update(band["labels"])
    return background_labels


def interval_coverage(intervals: Sequence[Tuple[int, int]]) -> int:
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return int(sum(end - start for start, end in merged))


def recover_stroke_mask_if_confident(mask: np.ndarray, strength: np.ndarray, lab: np.ndarray) -> np.ndarray:
    seed = mask.astype(np.uint8) > 0
    if int(seed.sum()) < 24:
        return mask.astype(np.uint8)

    quality = evaluate_mask_quality(seed.astype(np.uint8))
    seed_strength = strength[seed]
    expanded_seed = cv2.dilate(seed.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    background_strength = strength[~expanded_seed]
    if background_strength.size == 0:
        return mask.astype(np.uint8)

    signal_margin = float(np.median(seed_strength)) / max(1e-6, float(np.percentile(background_strength, 95)))
    seed_lab = lab[seed]
    seed_lab_std = float(np.mean(np.std(seed_lab, axis=0))) if len(seed_lab) > 1 else 0.0

    if signal_margin < 2.0:
        return mask.astype(np.uint8)
    if seed_lab_std > 20.0:
        return mask.astype(np.uint8)
    if quality["borderRatio"] > 0.03:
        return mask.astype(np.uint8)

    recovered = complete_stroke_from_seed(seed, strength, lab, color_limit=38.0, low_factor=0.28)
    if recovered is None:
        return mask.astype(np.uint8)
    recovered_quality = evaluate_mask_quality(recovered)
    if recovered_quality["borderRatio"] > 0.035:
        return mask.astype(np.uint8)
    if float(recovered.mean()) > 0.32 or float(recovered.mean()) > float(seed.mean()) * 3.0:
        return mask.astype(np.uint8)
    return recovered.astype(np.uint8)


def complete_stroke_from_seed(
    seed: np.ndarray,
    strength: np.ndarray,
    lab: np.ndarray,
    color_limit: float,
    low_factor: float,
) -> np.ndarray | None:
    seed_strength = strength[seed]
    if seed_strength.size < 24:
        return None

    low = max(1.0, float(np.percentile(seed_strength, 20)) * low_factor)
    seed_lab = np.median(lab[seed], axis=0)
    color_distance = np.linalg.norm(lab - seed_lab[None, None, :], axis=2)
    allowed = (strength > low) & (color_distance < color_limit)

    h, w = seed.shape
    yy, xx = np.mgrid[0:h, 0:w]
    allowed &= (np.abs(xx - w / 2) < w * 0.49) & (np.abs(yy - h / 2) < h * 0.49)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(allowed.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None

    keep_labels = set(int(label) for label in np.unique(labels[seed]) if label > 0)
    ys, xs = np.nonzero(seed)
    if len(xs) == 0:
        return None
    seed_box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    seed_area = int(seed.sum())

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        touches_edge = x <= 1 or y <= 1 or x + ww >= w - 2 or y + hh >= h - 2
        near_seed = expanded_boxes_intersect(seed_box, (x, y, x + ww, y + hh), padding=24)
        near_center = abs(cx - w / 2) < w * 0.40 and abs(cy - h / 2) < h * 0.40
        if touches_edge and area > seed_area * 0.35:
            continue
        if near_seed or near_center:
            keep_labels.add(label)

    if not keep_labels:
        return None

    recovered = np.isin(labels, list(keep_labels)).astype(np.uint8)
    recovered = clean_mask(recovered)
    recovered = suppress_background_texture_components(recovered)
    if float(recovered.mean()) <= float(seed.mean()) * 1.04:
        return None
    return recovered.astype(np.uint8)


def should_use_hysteresis_mask(seed_mask: np.ndarray, expanded_mask: np.ndarray) -> bool:
    seed_area = float(seed_mask.mean())
    expanded_area = float(expanded_mask.mean())
    if expanded_area < 0.006 or expanded_area > 0.24:
        return False
    if seed_area > 0 and expanded_area > seed_area * 2.75:
        return False
    quality = evaluate_mask_quality(expanded_mask)
    if quality["borderRatio"] > 0.04:
        return False
    if quality["components"] > 16:
        return False
    return True


def should_use_refined_mask(mask: np.ndarray, refined: np.ndarray) -> bool:
    original_quality = evaluate_mask_quality(mask)
    refined_quality = evaluate_mask_quality(refined)
    original_area = original_quality["area"]
    refined_area = refined_quality["area"]
    if refined_area < 0.006 or refined_area > original_area * 0.92:
        return False
    if original_quality["components"] > 10 or original_quality["borderRatio"] > 0.035:
        return True
    return refined_quality["penalty"] + 0.01 < original_quality["penalty"]


def adaptive_mask_score(quality: Dict[str, float], contrast: float, polarity: str) -> float:
    area = quality["area"]
    area_penalty = abs(area - 0.105) * 0.18
    contrast_bonus = min(0.035, contrast / 255.0 * 0.08)
    polarity_penalty = 0.001 if polarity == "light" else 0.0
    return quality["penalty"] + area_penalty + polarity_penalty - contrast_bonus


def add_threshold_pair(out: List[Tuple[str, np.ndarray]], name: str, values: np.ndarray, threshold: float) -> None:
    out.append((f"{name}-light", values > threshold))
    out.append((f"{name}-dark", values < threshold))


def add_distance_masks(out: List[Tuple[str, np.ndarray]], name: str, distance: np.ndarray) -> None:
    if float(distance.max()) <= 1e-6:
        return
    try:
        out.append((f"{name}-otsu", distance > threshold_otsu(distance)))
    except Exception:
        pass
    for percentile in (82, 88, 93):
        out.append((f"{name}-p{percentile}", distance > np.percentile(distance, percentile)))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.astype(np.uint8)
    out = cv2.medianBlur(out * 255, 3) > 0
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1) > 0
    out = remove_small_objects(out, min_size=10)
    return out.astype(np.uint8)


def refine_icon_components(mask: np.ndarray, rgb: np.ndarray, lab: np.ndarray) -> np.ndarray | None:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return None

    h, w = mask.shape
    center = np.array([w / 2, h / 2], dtype=np.float32)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_mask = labels == label
        centroid = np.asarray(centroids[label], dtype=np.float32)
        center_distance = float(np.linalg.norm((centroid - center) / max(w, h)))
        touches_border = x <= 1 or y <= 1 or x + ww >= w - 2 or y + hh >= h - 2
        median_lab = np.median(lab[component_mask], axis=0)
        seed_score = area * (1.0 - min(0.78, center_distance)) * (0.32 if touches_border else 1.0)
        components.append(
            {
                "label": label,
                "area": area,
                "bbox": (x, y, x + ww, y + hh),
                "centroid": centroid,
                "centerDistance": center_distance,
                "touchesBorder": touches_border,
                "medianLab": median_lab,
                "seedScore": seed_score,
            }
        )

    if len(components) <= 1:
        return None

    seed = max(components, key=lambda item: item["seedScore"])
    seed_color = seed["medianLab"]
    seed_bbox = seed["bbox"]
    keep_labels = set()
    for component in components:
        color_distance = float(np.linalg.norm(component["medianLab"] - seed_color))
        near_seed = expanded_boxes_intersect(seed_bbox, component["bbox"], padding=max(18, int(max(w, h) * 0.22)))
        center_ok = component["centerDistance"] < 0.52
        border_ok = not component["touchesBorder"] or component["area"] > seed["area"] * 0.45
        if color_distance < 38 and (near_seed or center_ok) and border_ok:
            keep_labels.add(component["label"])

    if seed["label"] not in keep_labels:
        keep_labels.add(seed["label"])

    refined = np.isin(labels, list(keep_labels)).astype(np.uint8)
    refined = clean_mask(refined)
    original_area = float(mask.mean())
    refined_area = float(refined.mean())
    if refined_area < 0.006 or refined_area >= original_area * 0.98:
        return None
    return refined


def expanded_boxes_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], padding: int) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ax0 -= padding
    ay0 -= padding
    ax1 += padding
    ay1 += padding
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def evaluate_mask_quality(mask: np.ndarray) -> Dict[str, float]:
    area = float(mask.mean())
    component_count, small_component_ratio = component_stats(mask)
    border_ratio = float(
        np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]).mean()
    )
    perimeter = mask_perimeter(mask)
    complexity = float(perimeter / max(1.0, math.sqrt(float(mask.sum()))))
    penalty = 0.0
    penalty += max(0.0, area - 0.24) * 0.16
    penalty += max(0.0, 0.012 - area) * 0.8
    penalty += max(0.0, border_ratio - 0.04) * 0.16
    penalty += max(0.0, component_count - 14) * 0.0014
    penalty += small_component_ratio * 0.012
    penalty += max(0.0, complexity - 28.0) * 0.0008
    return {
        "area": round(area, 5),
        "components": float(component_count),
        "smallComponentRatio": round(small_component_ratio, 5),
        "borderRatio": round(border_ratio, 5),
        "complexity": round(complexity, 5),
        "penalty": round(penalty, 5),
    }


def component_stats(mask: np.ndarray) -> Tuple[int, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    small = areas[areas < 18].sum()
    return int(count - 1), float(small / max(1.0, float(areas.sum())))


def mask_perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def mask_priority(name: str, mask: np.ndarray) -> Tuple[float, float]:
    area = float(mask.mean())
    quality = evaluate_mask_quality(mask)
    ideal_area_penalty = abs(area - 0.12) * 0.25
    name_bonus = 0.0
    if "border" in name:
        name_bonus -= 0.03
    if name.endswith("-icon"):
        name_bonus -= 0.035
    if name == "nn-gated-unet":
        name_bonus -= 0.16
    if "sauvola" in name:
        name_bonus += 0.06
    return (quality["penalty"] + ideal_area_penalty + name_bonus, area)


def potrace_variants(mask: np.ndarray) -> list[PotraceVariant]:
    base = preprocess_mask_for_potrace(mask)
    thickness = estimate_mask_thickness(base)
    variants = [
        PotraceVariant("potrace-default", base, potrace_options(thickness, "default"), 0.0),
        PotraceVariant("potrace-crisp", base, potrace_options(thickness, "crisp"), 0.002),
    ]

    compact = compact_mask_for_potrace(base)
    if not np.array_equal(compact, base):
        variants.append(PotraceVariant("potrace-compact", compact, potrace_options(thickness, "compact"), 0.004))

    if thickness >= 4.8:
        tight = shrink_mask_for_potrace(base)
        if int(tight.sum()) > max(16, int(base.sum() * 0.58)):
            variants.append(PotraceVariant("potrace-tight", tight, potrace_options(thickness, "tight"), 0.006))

    seen: set[bytes] = set()
    deduped: list[PotraceVariant] = []
    for variant in variants:
        key = cv2.resize(variant.mask.astype(np.uint8), (32, 32), interpolation=cv2.INTER_NEAREST).tobytes()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def preprocess_mask_for_potrace(mask: np.ndarray) -> np.ndarray:
    out = mask.astype(np.uint8)
    out = cv2.medianBlur(out * 255, 3) > 0
    out = remove_tiny_components(out.astype(np.uint8), min_area=5)
    out = fill_tiny_mask_holes(out, max_area=7)
    return out.astype(np.uint8)


def compact_mask_for_potrace(mask: np.ndarray) -> np.ndarray:
    out = mask.astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    out = remove_tiny_components(out, min_area=7)
    return fill_tiny_mask_holes(out, max_area=5).astype(np.uint8)


def shrink_mask_for_potrace(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    out = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    out = remove_tiny_components(out, min_area=5)
    return fill_tiny_mask_holes(out, max_area=5).astype(np.uint8)


def remove_tiny_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(np.uint8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out


def fill_tiny_mask_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    inv = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = mask.astype(np.uint8).copy()
    h, w = mask.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches_border and area <= max_area:
            out[labels == label] = 1
    return out


def estimate_mask_thickness(mask: np.ndarray) -> float:
    if int(mask.sum()) < 12:
        return 1.0
    centerline = skeletonize(mask.astype(bool))
    if int(centerline.sum()) < 4:
        return float(np.sqrt(max(1.0, float(mask.sum()))))
    distances = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    values = distances[centerline.astype(bool)]
    if values.size == 0:
        return 1.0
    return float(np.clip(np.median(values) * 2.0, 1.0, 18.0))


def potrace_options(thickness: float, mode: str) -> Dict[str, Any]:
    if mode == "crisp":
        return {"turdSize": 1, "optTolerance": 0.10, "alphaMax": 0.72}
    if mode == "compact":
        return {"turdSize": 2, "optTolerance": 0.16 if thickness < 5.5 else 0.22, "alphaMax": 0.86}
    if mode == "tight":
        return {"turdSize": 1, "optTolerance": 0.12, "alphaMax": 0.75}
    return {
        "turdSize": 2,
        "optTolerance": 0.18 if thickness < 5.0 else 0.26,
        "alphaMax": 0.95 if thickness < 5.0 else 1.12,
    }


def trace_with_potrace(mask_image: Image.Image, stroke_color: str, options: Dict[str, Any] | None = None) -> str:
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    script = """
	const fs = require('node:fs');
	const potrace = require('potrace');
	const color = process.argv[1];
	const extra = JSON.parse(process.argv[2] || '{}');
	const chunks = [];
	process.stdin.on('data', chunk => chunks.push(chunk));
	process.stdin.on('end', () => {
	  const input = Buffer.concat(chunks);
	  potrace.trace(input, {
	    threshold: 128,
	    blackOnWhite: true,
	    turdSize: 2,
	    optCurve: true,
	    optTolerance: 0.2,
	    alphaMax: 1.0,
	    ...extra,
	    color,
	    background: 'transparent'
	  }, (err, svg) => {
    if (err) {
      console.error(err.stack || String(err));
      process.exit(1);
    }
    process.stdout.write(svg);
  });
});
"""
    result = subprocess.run(
        ["node", "-e", script, stroke_color, json.dumps(options or {})],
        cwd=ROOT,
        input=buffer.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip() or "potrace failed")
    return result.stdout.decode("utf-8")


def mask_to_trace_bitmap(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((255 - mask.astype(np.uint8) * 255).astype(np.uint8)).convert("RGB")


def normalize_svg(svg_text: str, stroke_color: str, size: int) -> str:
    svg_text = re.sub(r"<\?xml[^>]*>\s*", "", svg_text).strip()
    svg_text = re.sub(r"<!--.*?-->\s*", "", svg_text, flags=re.S)
    svg_text = re.sub(r'fill="(?:#000000|#000|black)"', f'fill="{stroke_color}"', svg_text, flags=re.I)
    svg_text = re.sub(r'fill:\s*(?:#000000|#000|black)', f"fill:{stroke_color}", svg_text, flags=re.I)
    svg_text = re.sub(r'\s(width|height)="[^"]*"', "", svg_text, count=2)
    if "viewBox=" not in svg_text[:260]:
        svg_text = svg_text.replace("<svg ", f'<svg viewBox="0 0 {size} {size}" ', 1)
    svg_text = svg_text.replace("<svg ", f'<svg width="{size}" height="{size}" ', 1)
    return svg_text


def render_svg_transparent(svg_text: str, size: int) -> Image.Image:
    ensure_cairo_library_discovery()
    import cairosvg

    png = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=size,
        output_height=size,
    )
    return Image.open(io.BytesIO(png)).convert("RGBA")


def inpaint_background(source: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(source.convert("RGB"))
    inpaint_mask = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    restored = cv2.inpaint(bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))


def composite_rgba_over_rgb(foreground: Image.Image, background: Image.Image) -> Image.Image:
    canvas = background.convert("RGBA")
    canvas.alpha_composite(foreground.convert("RGBA"))
    return canvas.convert("RGB")


def transparent_preview(foreground: Image.Image, size: int) -> Image.Image:
    tile = 8
    preview = Image.new("RGB", (size, size), "#1a1a1a")
    draw = ImageDraw.Draw(preview)
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            fill = "#252525" if ((x // tile) + (y // tile)) % 2 == 0 else "#141414"
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=fill)
    canvas = preview.convert("RGBA")
    canvas.alpha_composite(foreground.convert("RGBA"))
    return canvas.convert("RGB")


def estimate_stroke_color(image: Image.Image, mask: np.ndarray) -> str:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_bool = mask.astype(bool)
    pixels = rgb[mask_bool]
    if pixels.size == 0:
        return "#000000"

    alpha, _ = constant_color_alpha_map(image)
    lab = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_bg = estimate_smooth_color_background_scaled(lab, 31, 8.0)
    residual = robust_unit_normalize(np.linalg.norm(lab - lab_bg, axis=2))
    distance = robust_unit_normalize(cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3))
    confidence = np.clip(alpha, 0.0, 1.0) * 0.56 + residual * 0.34 + distance * 0.10
    mask_confidence = confidence[mask_bool]

    if mask_confidence.size >= 12 and float(mask_confidence.max()) > 0:
        threshold = float(np.percentile(mask_confidence, 58))
        focused = rgb[mask_bool & (confidence >= threshold)]
    else:
        focused = pixels

    if focused.size == 0:
        focused = pixels

    gray = (focused[:, 0] * 0.299) + (focused[:, 1] * 0.587) + (focused[:, 2] * 0.114)
    lo, hi = np.percentile(gray, [6, 94])
    focused = focused[(gray >= lo) & (gray <= hi)]
    if focused.size == 0:
        focused = pixels
    return rgb_hex(np.median(focused, axis=0))


def estimate_border_color(image: Image.Image, border: int = 8) -> Tuple[int, int, int]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    pixels = border_pixels(rgb, border)
    return tuple(int(round(v)) for v in np.median(pixels, axis=0))


def border_pixels(arr: np.ndarray, border: int = 8) -> np.ndarray:
    return np.concatenate(
        [
            arr[:border, :, :].reshape(-1, arr.shape[2]),
            arr[-border:, :, :].reshape(-1, arr.shape[2]),
            arr[:, :border, :].reshape(-1, arr.shape[2]),
            arr[:, -border:, :].reshape(-1, arr.shape[2]),
        ],
        axis=0,
    )


def canonical_image(image: Image.Image, size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), estimate_border_color(image))
    scale = size / max(image.width, image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def mask_debug_image(mask: np.ndarray) -> Image.Image:
    image = Image.new("RGB", (mask.shape[1], mask.shape[0]), "#11130f")
    draw = ImageDraw.Draw(image)
    ys, xs = np.nonzero(mask)
    for x, y in zip(xs, ys):
        draw.point((int(x), int(y)), fill=(235, 225, 205))
    return image


def inline_svg_html(svg_text: str) -> str:
    return f'<span class="vector-icon">{svg_text}</span>'


def standalone_html(icon_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><style>
body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #171a14; }}
.vector-icon, .vector-icon svg {{ width: 128px; height: 128px; display: block; }}
</style></head>
<body>{icon_html}</body>
</html>
"""


def ensure_cairo_library_discovery() -> None:
    original_find_library = ctypes.util.find_library
    cairo_candidates = [
        Path("/opt/homebrew/opt/cairo/lib/libcairo.2.dylib"),
        Path("/opt/homebrew/lib/libcairo.2.dylib"),
        Path("/usr/local/opt/cairo/lib/libcairo.2.dylib"),
        Path("/usr/local/lib/libcairo.2.dylib"),
    ]

    def patched_find_library(name: str) -> str | None:
        if name in {"cairo-2", "cairo", "libcairo-2"}:
            for candidate in cairo_candidates:
                if candidate.exists():
                    return str(candidate)
        return original_find_library(name)

    ctypes.util.find_library = patched_find_library


def rgb_hex(values: Sequence[float]) -> str:
    return "#" + "".join(f"{int(max(0, min(255, round(v)))):02x}" for v in values)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
