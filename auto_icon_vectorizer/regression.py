"""Regression sheet for the production icon vectorizer selector.

This is intentionally CLI-only. It exercises the same adapter used by the
package and writes a visual comparison sheet plus JSON metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from auto_icon_vectorizer import vectorize as adapter
else:
    from . import vectorize as adapter

JsonDict = Dict[str, Any]
SIZE = 128
SCALE = 4


def main() -> None:
    warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
    parser = argparse.ArgumentParser(description="Run icon vectorizer routing regressions.")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/regression-output"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = adapter._load_runtime()
    cases = _cases()
    rows: List[JsonDict] = []
    panels = []

    for case in cases:
        source, truth = case["build"]()
        canonical = trace.canonical_image(source, SIZE)
        auto = adapter._select_candidate(trace, canonical, "auto")
        stem = args.output_dir / case["id"]
        result = adapter.vectorize_icon_crop(
            source,
            source_id=case["id"],
            class_name="regression-icon",
            output_prefix=stem,
            mask_mode="auto",
        )

        row: JsonDict = {
            "id": case["id"],
            "kind": case["kind"],
            "expectedMode": case["expected_mode"],
            "selectedMode": auto["mode"],
            "renderer": result["diagnostics"]["renderer"],
            "foregroundRatio": round(float(auto["mask"].mean()), 6),
            "autoDecision": auto.get("auto_decision"),
        }
        if truth is not None:
            row.update(_mask_metrics(auto["mask"], truth))
        row["passed"] = _passes(case, row)
        rows.append(row)
        panels.append((case, source, truth, auto, row))

    summary = {
        "passed": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "bySelectedMode": _count(row["selectedMode"] for row in rows),
    }
    payload = {"summary": summary, "rows": rows}
    (args.output_dir / "icon-vectorizer-regression.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sheet = _write_sheet(trace, panels, summary, args.output_dir / "icon-vectorizer-regression.png")
    print(sheet)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


def _cases() -> List[JsonDict]:
    return [
        {"id": "hybrid_pin_route", "kind": "hybrid", "expected_mode": "filled", "min_iou": 0.86, "build": _hybrid_pin_route},
        {"id": "hybrid_tag_string", "kind": "hybrid", "expected_mode": "filled", "min_iou": 0.84, "build": _hybrid_tag_string},
        {"id": "hybrid_star_orbit", "kind": "hybrid", "expected_mode": "any", "min_iou": 0.82, "build": _hybrid_star_orbit},
        {"id": "outline_cube", "kind": "outline", "expected_mode": "stroke", "build": _outline_cube},
        {"id": "outline_check_circle", "kind": "outline", "expected_mode": "stroke", "build": _outline_check_circle},
        {"id": "outline_chat_slash", "kind": "outline", "expected_mode": "stroke", "build": _outline_chat_slash},
    ]


def _passes(case: JsonDict, row: JsonDict) -> bool:
    if case["expected_mode"] != "any" and row["selectedMode"] != case["expected_mode"]:
        return False
    if "min_iou" in case and float(row.get("iou", 0.0)) < float(case["min_iou"]):
        return False
    return True


def _count(values: Any) -> JsonDict:
    counts: JsonDict = {}
    for value in values:
        counts[str(value)] = int(counts.get(str(value), 0)) + 1
    return counts


def _canvas(bg: Tuple[int, int, int] = (18, 20, 17)) -> Image.Image:
    return Image.new("RGB", (SIZE * SCALE, SIZE * SCALE), bg)


def _mask() -> Image.Image:
    return Image.new("L", (SIZE * SCALE, SIZE * SCALE), 0)


def _down_rgb(image: Image.Image) -> Image.Image:
    return image.resize((SIZE, SIZE), Image.Resampling.LANCZOS).convert("RGB")


def _down_mask(mask: Image.Image) -> np.ndarray:
    return (np.asarray(mask.resize((SIZE, SIZE), Image.Resampling.LANCZOS)) > 20).astype(np.uint8)


def _draw_both(draw_fn: Callable[[ImageDraw.ImageDraw, Tuple[int, int, int] | int], None], bg: Tuple[int, int, int], color: Tuple[int, int, int]) -> tuple[Image.Image, np.ndarray]:
    src = _canvas(bg)
    truth = _mask()
    draw_fn(ImageDraw.Draw(src), color)
    draw_fn(ImageDraw.Draw(truth), 255)
    return _down_rgb(src), _down_mask(truth)


def _hybrid_pin_route() -> tuple[Image.Image, np.ndarray]:
    def draw(drawer: ImageDraw.ImageDraw, color: Tuple[int, int, int] | int) -> None:
        s = SCALE
        drawer.ellipse([43*s, 14*s, 86*s, 62*s], fill=color)
        drawer.polygon([(51*s, 51*s), (78*s, 51*s), (65*s, 105*s)], fill=color)
        drawer.ellipse([57*s, 25*s, 73*s, 41*s], fill=0 if isinstance(color, int) else (18, 20, 17))
        drawer.line([(18*s, 94*s), (34*s, 78*s), (51*s, 92*s), (70*s, 80*s), (96*s, 92*s)], fill=color, width=7*s, joint="curve")

    return _draw_both(draw, (17, 20, 16), (204, 72, 43))


def _hybrid_tag_string() -> tuple[Image.Image, np.ndarray]:
    def draw(drawer: ImageDraw.ImageDraw, color: Tuple[int, int, int] | int) -> None:
        s = SCALE
        drawer.polygon([(20*s, 55*s), (61*s, 24*s), (106*s, 34*s), (95*s, 78*s), (55*s, 108*s)], fill=color)
        drawer.ellipse([75*s, 42*s, 88*s, 55*s], fill=0 if isinstance(color, int) else (75, 82, 64))
        drawer.arc([76*s, 12*s, 119*s, 56*s], 205, 34, fill=color, width=6*s)
        drawer.line([(107*s, 27*s), (116*s, 17*s)], fill=color, width=5*s)

    src, truth = _draw_both(draw, (75, 82, 64), (66, 125, 89))
    bg = ImageDraw.Draw(src)
    for x, y, r, c in [
        (18, 29, 7, (155, 119, 33)),
        (106, 12, 5, (20, 117, 142)),
        (14, 89, 8, (140, 70, 155)),
        (111, 95, 3, (191, 68, 52)),
        (34, 113, 5, (162, 130, 40)),
    ]:
        bg.ellipse([x-r, y-r, x+r, y+r], fill=c)
    return src, truth


def _hybrid_star_orbit() -> tuple[Image.Image, np.ndarray]:
    def star_points(cx: float, cy: float, outer: float, inner: float) -> list[tuple[float, float]]:
        pts = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            radius = outer if index % 2 == 0 else inner
            pts.append(((cx + math.cos(angle) * radius) * SCALE, (cy + math.sin(angle) * radius) * SCALE))
        return pts

    def draw(drawer: ImageDraw.ImageDraw, color: Tuple[int, int, int] | int) -> None:
        s = SCALE
        drawer.polygon(star_points(64, 73, 31, 13), fill=color)
        drawer.arc([19*s, 26*s, 109*s, 91*s], 196, 344, fill=color, width=7*s)

    return _draw_both(draw, (246, 241, 228), (189, 151, 74))


def _outline_cube() -> tuple[Image.Image, None]:
    def draw(drawer: ImageDraw.ImageDraw, color: Tuple[int, int, int] | int) -> None:
        s = SCALE
        pts = [(31*s, 38*s), (64*s, 20*s), (98*s, 38*s), (98*s, 82*s), (64*s, 105*s), (30*s, 82*s)]
        drawer.line(pts + [pts[0]], fill=color, width=5*s, joint="curve")
        drawer.line([(31*s, 38*s), (64*s, 57*s), (98*s, 38*s)], fill=color, width=5*s)
        drawer.line([(64*s, 57*s), (64*s, 105*s)], fill=color, width=5*s)
        drawer.line([(47*s, 29*s), (81*s, 48*s)], fill=color, width=5*s)

    src, _truth = _draw_both(draw, (19, 21, 18), (202, 184, 143))
    return src, None


def _outline_check_circle() -> tuple[Image.Image, None]:
    def draw(drawer: ImageDraw.ImageDraw, color: Tuple[int, int, int] | int) -> None:
        s = SCALE
        drawer.ellipse([23*s, 16*s, 105*s, 98*s], outline=color, width=5*s)
        drawer.line([(44*s, 57*s), (57*s, 70*s), (84*s, 40*s)], fill=color, width=7*s, joint="curve")

    src, _truth = _draw_both(draw, (20, 22, 18), (203, 184, 144))
    return src, None


def _outline_chat_slash() -> tuple[Image.Image, None]:
    def draw(drawer: ImageDraw.ImageDraw, color: Tuple[int, int, int] | int) -> None:
        s = SCALE
        drawer.ellipse([14*s, 14*s, 114*s, 114*s], outline=color, width=5*s)
        drawer.rectangle([39*s, 43*s, 88*s, 75*s], outline=color, width=5*s)
        drawer.line([(47*s, 75*s), (38*s, 86*s)], fill=color, width=5*s)
        drawer.line([(30*s, 29*s), (99*s, 99*s)], fill=color, width=6*s)

    src, _truth = _draw_both(draw, (19, 21, 18), (203, 184, 144))
    return src, None


def _mask_metrics(mask: np.ndarray, truth: np.ndarray) -> JsonDict:
    pred = mask.astype(bool)
    target = truth.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "iou": round(tp / max(1, union), 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _render_preview(trace: Any, candidate: JsonDict) -> Image.Image:
    return trace.transparent_preview(trace.render_svg_transparent(candidate["svg"], SIZE), SIZE).convert("RGB")


def _truth_preview(truth: np.ndarray | None) -> Image.Image:
    if truth is None:
        return Image.new("RGB", (SIZE, SIZE), "#f6f3ec")
    gray = np.where(truth.astype(bool), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([gray, gray, gray]))


def _write_sheet(trace: Any, panels: list[tuple[JsonDict, Image.Image, np.ndarray | None, JsonDict, JsonDict]], summary: JsonDict, path: Path) -> Path:
    panel = 128
    gap = 12
    left = 230
    header = 42
    row_h = panel + 64
    cols = ["source", "truth", "auto svg", "auto mask"]
    canvas = Image.new("RGB", (left + len(cols) * panel + (len(cols) + 1) * gap, header + len(panels) * row_h + gap), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), f"icon vectorizer regression: {summary['passed']}/{summary['total']} passed", fill="#222", font=font)
    x = left + gap
    for col in cols:
        draw.text((x, 14), col, fill="#222", font=font)
        x += panel + gap
    for index, (case, source, truth, auto, row) in enumerate(panels):
        y = header + index * row_h + gap
        draw.text((12, y + 2), str(case["id"]), fill="#111", font=font)
        draw.text((12, y + 18), f"expected={case['expected_mode']} selected={row['selectedMode']}", fill="#444", font=font)
        if truth is not None:
            draw.text((12, y + 34), f"IoU={row['iou']:.3f} F1={row['f1']:.3f}", fill="#444", font=font)
        draw.text((12, y + 50), "PASS" if row["passed"] else "FAIL", fill="#2f7d4f" if row["passed"] else "#b94a38", font=font)
        images = [
            source,
            _truth_preview(truth),
            _render_preview(trace, auto),
            trace.mask_debug_image(auto["mask"]).convert("RGB"),
        ]
        x = left + gap
        for image in images:
            draw.rectangle((x - 1, y - 1, x + panel, y + panel), outline="#d0cdc4")
            canvas.paste(image.resize((panel, panel), Image.Resampling.NEAREST), (x, y))
            x += panel + gap
    canvas.save(path)
    return path


if __name__ == "__main__":
    main()
