"""Auto stroke/filled icon vectorizer.

Pass an already-cropped raster icon image to this module. It returns HTML
containing an inline SVG, the raw SVG, simple path metadata, and
diagnostics describing the selected mask branch.
"""

from __future__ import annotations

import argparse
import html as html_lib
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

from PIL import Image

JsonDict = Dict[str, Any]

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
STROKE_RENDERER = "nn-gated-unet+potrace-default"
FILLED_RENDERER = "filled-silhouette-unet+potrace-default"
AUTO_RENDERER = "auto-stroke-filled+potrace-default"
RENDERER = AUTO_RENDERER
MASK_STRATEGY = "nn-gated-unet"
TRACE_VARIANT = "potrace-default"
FILLED_MODEL = RUNTIME_DIR / "nn-seg-results" / "best-filled-silhouette-unet.pt"
FILLED_MIN_AREA_RATIO = 1.25


def vectorize_icon_crop(
    crop: Image.Image,
    *,
    source_id: str | None = None,
    node_id: str | None = None,
    class_name: str = "vector-icon",
    title: str | None = None,
    aria_label: str | None = None,
    output_prefix: Path | None = None,
    mask_mode: str = "auto",
) -> JsonDict:
    """Convert an icon crop to inline SVG HTML using the local vendored runtime."""

    trace = _load_runtime()
    size = int(trace.SIZE)
    source = trace.canonical_image(crop.convert("RGB"), size)
    selected = _select_candidate(trace, source, mask_mode)
    mask = selected["mask"]
    svg = selected["svg"]
    svg = _prepare_svg(svg, title=title, aria_label=aria_label)
    html = _wrap_svg(svg, class_name=class_name, source_id=source_id or node_id, renderer=selected["renderer"])
    artifacts = _write_artifacts(trace, source, mask, svg, html, Path(output_prefix)) if output_prefix else {}
    paths = [
        {
            "type": "potrace_path",
            "pathCount": int(svg.count("<path")),
            "renderer": selected["renderer"],
        }
    ]

    return {
        "html": html,
        "svg": svg,
        "paths": paths,
        "primitives": paths,
        "diagnostics": {
            "pipelineRenderer": AUTO_RENDERER if mask_mode == "auto" else selected["renderer"],
            "renderer": selected["renderer"],
            "requestedMaskMode": mask_mode,
            "selectedMaskMode": selected["mode"],
            "maskStrategy": selected["mask_strategy"],
            "traceVariant": TRACE_VARIANT,
            "inputWidth": int(crop.width),
            "inputHeight": int(crop.height),
            "canonicalSize": size,
            "foregroundPixels": int(mask.sum()),
            "rawForegroundPixels": int(selected["raw_mask"].sum()),
            "foregroundRatio": round(float(mask.mean()), 6),
            "maskThickness": round(float(selected["thickness"]), 4),
            "strokeColor": selected["stroke_color"],
            "potraceOptions": selected["potrace_options"],
            "pathCount": int(svg.count("<path")),
            "selectionScore": selected.get("selection_score"),
            "selectionMetrics": selected.get("selection_metrics"),
            "candidateScores": selected.get("candidate_scores", []),
            "autoDecision": selected.get("auto_decision"),
            "runtimeDir": str(RUNTIME_DIR),
            "artifacts": artifacts,
        },
    }


def runtime_status() -> JsonDict:
    trace = _load_runtime()
    return {
        "renderer": RENDERER,
        "runtimeDir": str(RUNTIME_DIR),
        "modelPath": str(trace.NN_GATED_UNET_MODEL),
        "modelPresent": bool(trace.NN_GATED_UNET_MODEL.exists()),
        "filledModelPath": str(FILLED_MODEL),
        "filledModelPresent": bool(FILLED_MODEL.exists()),
        "potracePackagePresent": bool((RUNTIME_DIR / "node_modules" / "potrace").exists()),
        "cairoVendorPresent": bool((RUNTIME_DIR / ".vendor" / "python" / "cairosvg").exists()),
    }


def _load_runtime() -> Any:
    existing = sys.modules.get("trace_icon_component")
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path.parent == RUNTIME_DIR:
            return existing
        for module_name in [
            "trace_icon_component",
            "train_aux_fusion_icon_segmenter",
            "train_filled_silhouette_segmenter",
            "generate_spectral_evidence_bank",
            "apply_svm_connections",
        ]:
            sys.modules.pop(module_name, None)

    runtime_path = str(RUNTIME_DIR)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    return importlib.import_module("trace_icon_component")


def _prepare_svg(svg: str, *, title: str | None, aria_label: str | None) -> str:
    svg = _ensure_svg_attr(svg, "preserveAspectRatio", "xMidYMid meet")
    svg = _ensure_svg_attr(svg, "focusable", "false")
    if aria_label or title:
        label = aria_label or title or ""
        svg = _ensure_svg_attr(svg, "role", "img")
        svg = _ensure_svg_attr(svg, "aria-label", label)
    else:
        svg = _ensure_svg_attr(svg, "aria-hidden", "true")
    if title:
        title_tag = f"<title>{html_lib.escape(title)}</title>"
        svg = re.sub(r"(<svg\b[^>]*>)", r"\1" + title_tag, svg, count=1)
    return svg


def _ensure_svg_attr(svg: str, name: str, value: str) -> str:
    if re.search(rf"\s{name}=", svg[:320]):
        return svg
    escaped = html_lib.escape(value, quote=True)
    return re.sub(r"<svg\b", f'<svg {name}="{escaped}"', svg, count=1)


def _select_candidate(trace: Any, source: Image.Image, mask_mode: str) -> JsonDict:
    mode = mask_mode.lower().strip()
    if mode not in {"auto", "stroke", "filled"}:
        raise ValueError("mask_mode must be one of: auto, stroke, filled")

    candidates = []
    if mode in {"auto", "stroke"}:
        try:
            mask_name, raw_mask = trace.fixed_mask_candidate(source, MASK_STRATEGY)
            candidates.append(_trace_candidate(trace, source, "stroke", STROKE_RENDERER, str(mask_name), raw_mask))
        except Exception:
            if mode == "stroke":
                raise
    if mode in {"auto", "filled"}:
        raw_mask = _filled_silhouette_mask(source)
        if raw_mask is None:
            if mode == "filled":
                raise RuntimeError("filled silhouette model is not trained yet")
        else:
            try:
                candidates.append(_trace_candidate(trace, source, "filled", FILLED_RENDERER, "filled-silhouette-unet", raw_mask))
            except Exception:
                if mode == "filled":
                    raise

    if not candidates:
        raise RuntimeError("No mask candidate produced a traceable SVG")

    for candidate in candidates:
        candidate["selection_score"] = round(_score_candidate(trace, source, candidate), 6)

    if mode == "auto":
        selected = _auto_select(candidates)
    else:
        candidates.sort(key=lambda item: item["selection_score"])
        selected = candidates[0]
    selected["candidate_scores"] = [
        {
            "mode": item["mode"],
            "renderer": item["renderer"],
            "maskStrategy": item["mask_strategy"],
            "foregroundRatio": round(float(item["mask"].mean()), 6),
            "selectionScore": item["selection_score"],
            "metrics": item.get("selection_metrics"),
        }
        for item in candidates
    ]
    return selected


def _auto_select(candidates: list[JsonDict]) -> JsonDict:
    by_mode = {str(candidate["mode"]): candidate for candidate in candidates}
    stroke = by_mode.get("stroke")
    filled = by_mode.get("filled")
    if stroke is None or filled is None:
        candidates.sort(key=lambda item: item["selection_score"])
        selected = candidates[0]
        selected["auto_decision"] = "only_available_candidate"
        return selected

    stroke_area = float(stroke["mask"].mean())
    filled_area = float(filled["mask"].mean())
    stroke_score = float(stroke["selection_score"])
    filled_score = float(filled["selection_score"])
    area_ratio = filled_area / max(1e-6, stroke_area)
    filled_is_plausible = 0.075 <= filled_area <= 0.55
    filled_substantially_larger = filled_area >= max(0.115, stroke_area * FILLED_MIN_AREA_RATIO)
    filled_score_close = filled_score <= stroke_score + 0.012

    if filled_is_plausible and filled_substantially_larger and filled_score_close:
        filled["auto_decision"] = (
            f"filled_area_ratio={area_ratio:.2f}; filled score close enough and mask is materially larger"
        )
        return filled

    stroke["auto_decision"] = (
        f"stroke_default; filled_area_ratio={area_ratio:.2f}; "
        f"filled_area={filled_area:.3f}; stroke_area={stroke_area:.3f}"
    )
    return stroke


def _trace_candidate(
    trace: Any,
    source: Image.Image,
    mode: str,
    renderer: str,
    mask_strategy: str,
    raw_mask: Any,
) -> JsonDict:
    size = int(trace.SIZE)
    mask = trace.preprocess_mask_for_potrace(raw_mask)
    if int(mask.sum()) <= 0:
        raise RuntimeError(f"{mask_strategy} produced an empty mask")
    thickness = float(trace.estimate_mask_thickness(mask))
    potrace_options = trace.potrace_options(thickness, "default")
    stroke_color = trace.estimate_stroke_color(source, mask)
    svg = trace.normalize_svg(
        trace.trace_with_potrace(trace.mask_to_trace_bitmap(mask), stroke_color, potrace_options),
        stroke_color,
        size,
    )
    return {
        "mode": mode,
        "renderer": renderer,
        "mask_strategy": mask_strategy,
        "raw_mask": raw_mask,
        "mask": mask,
        "thickness": thickness,
        "potrace_options": potrace_options,
        "stroke_color": stroke_color,
        "svg": svg,
    }


def _score_candidate(trace: Any, source: Image.Image, candidate: JsonDict) -> float:
    try:
        rendered = trace.render_svg_transparent(candidate["svg"], int(trace.SIZE))
        background = trace.inpaint_background(source, candidate["mask"])
        composite = trace.composite_rgba_over_rgb(rendered, background)
        metrics = trace.score_pair(source, composite)
        candidate["selection_metrics"] = {
            key: round(float(value), 6) for key, value in metrics.items() if isinstance(value, (int, float))
        }
        score = float(metrics["priority_score"])
    except Exception as exc:
        candidate["selection_metrics"] = {"error": str(exc)}
        score = 999.0

    area = float(candidate["mask"].mean())
    if candidate["mode"] == "filled":
        if area < 0.045:
            score += 0.12
        if area > 0.62:
            score += 0.18
    else:
        if area > 0.34:
            score += 0.08
    return score


def _filled_silhouette_mask(source: Image.Image) -> Any | None:
    try:
        from train_filled_silhouette_segmenter import filled_silhouette_unet_mask

        return filled_silhouette_unet_mask(source)
    except Exception:
        return None


def _wrap_svg(svg: str, *, class_name: str, source_id: str | None, renderer: str) -> str:
    classes = _normalize_classes(class_name)
    attrs = [
        f'class="{html_lib.escape(classes, quote=True)}"',
        f'data-vectorizer="{html_lib.escape(renderer, quote=True)}"',
    ]
    if source_id:
        attrs.append(f'data-source-id="{html_lib.escape(source_id, quote=True)}"')
    return f"<span {' '.join(attrs)}>{svg}</span>"


def _normalize_classes(class_name: str) -> str:
    names = [name for name in re.split(r"\s+", class_name.strip()) if name]
    if "vector-icon" not in names:
        names.insert(0, "vector-icon")
    return " ".join(dict.fromkeys(names))


def _write_artifacts(trace: Any, source: Image.Image, mask: Any, svg: str, icon_html: str, output_prefix: Path) -> JsonDict:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "source": str(output_prefix.with_name(output_prefix.name + "-source.png")),
        "mask": str(output_prefix.with_name(output_prefix.name + "-mask.png")),
        "svg": str(output_prefix.with_suffix(".svg")),
        "html": str(output_prefix.with_suffix(".html")),
    }
    source.save(artifacts["source"])
    trace.mask_debug_image(mask).save(artifacts["mask"])
    Path(artifacts["svg"]).write_text(svg, encoding="utf-8")
    Path(artifacts["html"]).write_text(trace.standalone_html(icon_html), encoding="utf-8")

    try:
        rendered = trace.transparent_preview(trace.render_svg_transparent(svg, int(trace.SIZE)), int(trace.SIZE))
        rendered_path = output_prefix.with_name(output_prefix.name + "-rendered.png")
        rendered.save(rendered_path)
        artifacts["rendered"] = str(rendered_path)
    except Exception as exc:
        artifacts["renderedError"] = str(exc)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize one icon crop with auto-stroke-filled+potrace-default.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out-prefix", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--node-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--class-name", default="vector-icon")
    parser.add_argument("--title", default=None)
    parser.add_argument("--aria-label", default=None)
    parser.add_argument("--mask-mode", choices=["auto", "stroke", "filled"], default="auto")
    args = parser.parse_args()

    crop = Image.open(args.image).convert("RGB")
    result = vectorize_icon_crop(
        crop,
        source_id=args.source_id or args.node_id,
        class_name=args.class_name,
        title=args.title,
        aria_label=args.aria_label,
        output_prefix=args.out_prefix,
        mask_mode=args.mask_mode,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
