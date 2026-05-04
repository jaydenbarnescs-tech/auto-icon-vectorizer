"""Apply the trained SVM connection model to vectorizer fixture output.

Input is JSON on stdin:
{
  "original": "...png",
  "outputDir": "...",
  "fixtureNumber": 1,
  "size": 128,
  "backgroundColor": "#...",
  "strokeColor": "#...",
  "strokeWidth": 3.82,
  "primitives": [...]
}

The script proposes endpoint connections using the trained SVM and accepts only
connections that improve the same Pillow visual-diff score used by the fixture
evaluator.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image, ImageDraw


MODEL_PATH = Path("svm-connection-model/endpoint_connection_svm.joblib")


def main() -> None:
    payload = json.loads(sys.stdin.read())
    model = joblib.load(MODEL_PATH)
    result = apply_svm_connections(payload, model)
    print(json.dumps(result, ensure_ascii=False))


def apply_svm_connections(payload: Dict[str, Any], model: Any) -> Dict[str, Any]:
    size = int(payload["size"])
    primitives = clone_primitives(payload["primitives"])
    bg = hex_to_rgb(payload["backgroundColor"])
    stroke = hex_to_rgb(payload["strokeColor"])
    stroke_width = float(payload["strokeWidth"])
    original = canonical_original(payload["original"], size, bg)

    before_metrics = score_pair(original, render_prediction(primitives, size, bg, stroke, stroke_width))
    current_primitives = primitives
    current_metrics = before_metrics
    accepted = []

    proposals = svm_proposals(current_primitives, model, size)
    for proposal in proposals[:16]:
        candidate = clone_primitives(current_primitives)
        candidate.append({
            "type": "line",
            "x1": round(proposal["a"]["x"], 2),
            "y1": round(proposal["a"]["y"], 2),
            "x2": round(proposal["b"]["x"], 2),
            "y2": round(proposal["b"]["y"], 2),
            "error": 0,
            "source": "svm_connection",
        })
        if duplicate_line(candidate[-1], current_primitives):
            continue
        metrics = score_pair(original, render_prediction(candidate, size, bg, stroke, stroke_width))
        # Use a real threshold because tiny improvements do not survive renderer differences.
        if metrics["priority_score"] + 0.0025 < current_metrics["priority_score"]:
            current_primitives = candidate
            current_metrics = metrics
            accepted.append({**proposal, "metrics": metrics})

    artifacts = write_artifacts(payload, original, current_primitives, bg, stroke, stroke_width, current_metrics)
    return {
        "primitives": current_primitives,
        "metrics": current_metrics,
        "beforeMetrics": before_metrics,
        "acceptedConnections": accepted,
        "proposalCount": len(proposals),
        "artifacts": artifacts,
    }


def svm_proposals(primitives: Sequence[Dict[str, Any]], model: Any, size: int) -> List[Dict[str, Any]]:
    endpoints = line_endpoints(primitives)
    rows = []
    pairs = []
    for i, a in enumerate(endpoints):
        nearest = []
        for j, b in enumerate(endpoints):
            if i == j:
                continue
            d = distance(a["point"], b["point"])
            if d <= 28:
                nearest.append((d, j))
        nearest.sort()
        for _, j in nearest[:12]:
            if j <= i:
                continue
            b = endpoints[j]
            rows.append(features_for_pair(a, b, endpoints))
            pairs.append((a, b))
    if not rows:
        return []

    x = np.asarray(rows, dtype=np.float32)
    pred = model.predict(x)
    scores = model.decision_function(x)
    proposals = []
    for (a, b), label, score in zip(pairs, pred, scores):
        if int(label) != 1:
            continue
        if float(score) < 0.15:
            continue
        if same_line(a, b):
            continue
        proposals.append({
            "score": float(score),
            "distance": distance(a["point"], b["point"]),
            "a": a["point"],
            "b": b["point"],
            "aLine": a["primitiveIndex"],
            "bLine": b["primitiveIndex"],
        })
    proposals.sort(key=lambda row: (-row["score"], row["distance"]))
    return proposals


def line_endpoints(primitives: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    endpoints = []
    for index, primitive in enumerate(primitives):
        if primitive.get("type") != "line":
            continue
        a = {"x": float(primitive["x1"]), "y": float(primitive["y1"])}
        b = {"x": float(primitive["x2"]), "y": float(primitive["y2"])}
        endpoints.append({"point": a, "other": b, "primitiveIndex": index, "endpoint": "a"})
        endpoints.append({"point": b, "other": a, "primitiveIndex": index, "endpoint": "b"})
    return endpoints


def features_for_pair(a: Dict[str, Any], b: Dict[str, Any], endpoints: Sequence[Dict[str, Any]]) -> List[float]:
    da = direction(a["other"], a["point"])
    db = direction(b["other"], b["point"])
    connector = direction(a["point"], b["point"])
    endpoint_distance = distance(a["point"], b["point"])
    angle_diff = abs_angle_diff(da, db)
    continuation_angle = min(abs_angle_diff(da, connector), abs_angle_diff(db, connector + math.pi))
    approach_angle_a = abs_angle_diff(da, connector)
    approach_angle_b = abs_angle_diff(db, connector + math.pi)
    intersection = ray_intersection(a["point"], da, b["point"], db)
    intersection_distance = 128.0 if intersection is None else min(
        128.0,
        distance(a["point"], intersection) + distance(b["point"], intersection),
    )
    length_a = distance(a["point"], a["other"])
    length_b = distance(b["point"], b["other"])
    length_ratio = min(length_a, length_b) / max(length_a, length_b, 1e-6)
    nearest_third_endpoint = nearest_other_endpoint_distance(a, b, endpoints)
    return [
        endpoint_distance / 128.0,
        angle_diff / math.pi,
        continuation_angle / math.pi,
        approach_angle_a / math.pi,
        approach_angle_b / math.pi,
        intersection_distance / 128.0,
        length_ratio,
        nearest_third_endpoint / 128.0,
        math.cos(da - db),
        math.sin(da - db),
        math.cos(connector - da),
        math.sin(connector - da),
    ]


def canonical_original(path: str, size: int, bg: Tuple[int, int, int]) -> Image.Image:
    src = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (size, size), bg)
    scale = size / max(src.width, src.height)
    new_size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    resized = src.resize(new_size, Image.Resampling.LANCZOS)
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def render_prediction(primitives: Sequence[Dict[str, Any]], size: int, bg: Tuple[int, int, int], stroke: Tuple[int, int, int], stroke_width: float) -> Image.Image:
    scale = 4
    image = Image.new("RGB", (size * scale, size * scale), bg)
    draw = ImageDraw.Draw(image)
    width = max(1, round(stroke_width * scale))
    for primitive in primitives:
        if primitive["type"] == "circle":
            cx = primitive["cx"] * scale
            cy = primitive["cy"] * scale
            r = primitive["r"] * scale
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=stroke, width=width)
        elif primitive["type"] == "line":
            draw.line((primitive["x1"] * scale, primitive["y1"] * scale, primitive["x2"] * scale, primitive["y2"] * scale), fill=stroke, width=width)
        elif primitive["type"] == "polyline":
            points = [(p["x"] * scale, p["y"] * scale) for p in primitive["points"]]
            if len(points) >= 2:
                draw.line(points, fill=stroke, width=width)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def score_pair(original: Image.Image, rendered: Image.Image) -> Dict[str, float]:
    orig_arr = np.asarray(original.convert("RGB"), dtype=np.float32)
    rend_arr = np.asarray(rendered.convert("RGB"), dtype=np.float32)
    diff = np.abs(orig_arr - rend_arr)
    mae = float(diff.mean())
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    diff_gray = diff.mean(axis=2)
    high_diff_ratio = float((diff_gray > 40).mean())
    orig_gray = gray(orig_arr)
    rend_gray = gray(rend_arr)
    edge_mae = float(np.abs(edge_map(orig_gray) - edge_map(rend_gray)).mean() / 255.0)
    dark_delta = float(abs((orig_gray < 96).mean() - (rend_gray < 96).mean()))
    mae_norm = mae / 255.0
    priority = min(1.0, (mae_norm * 0.48) + (edge_mae * 0.34) + (high_diff_ratio * 0.14) + (dark_delta * 0.04))
    return {
        "mae_rgb": round(mae, 4),
        "mae_norm": round(mae_norm, 5),
        "rmse_rgb": round(rmse, 4),
        "edge_mae_norm": round(edge_mae, 5),
        "high_diff_ratio": round(high_diff_ratio, 5),
        "dark_pixel_ratio_delta": round(dark_delta, 5),
        "priority_score": round(priority, 5),
    }


def write_artifacts(payload: Dict[str, Any], original: Image.Image, primitives: Sequence[Dict[str, Any]], bg: Tuple[int, int, int], stroke: Tuple[int, int, int], stroke_width: float, metrics: Dict[str, float]) -> Dict[str, str]:
    out_dir = Path(payload["outputDir"])
    prefix = out_dir / f"fixture-{payload['fixtureNumber']}"
    rendered = render_prediction(primitives, int(payload["size"]), bg, stroke, stroke_width)
    heat = diff_heatmap(original, rendered)
    overlay = Image.blend(original, heat, 0.38)
    original_path = str(prefix) + "-canonical-original.png"
    rendered_path = str(prefix) + "-rendered.png"
    heat_path = str(prefix) + "-diff-heatmap.png"
    overlay_path = str(prefix) + "-diff-overlay.png"
    original.save(original_path)
    rendered.save(rendered_path)
    heat.save(heat_path)
    overlay.save(overlay_path)
    return {
        "canonicalOriginal": original_path,
        "rendered": rendered_path,
        "diffHeatmap": heat_path,
        "diffOverlay": overlay_path,
    }


def diff_heatmap(original: Image.Image, rendered: Image.Image) -> Image.Image:
    orig = np.asarray(original.convert("RGB"), dtype=np.float32)
    rend = np.asarray(rendered.convert("RGB"), dtype=np.float32)
    diff = np.abs(orig - rend).mean(axis=2)
    scaled = np.clip(diff * 3.2, 0, 255).astype(np.uint8)
    heat = np.zeros((*scaled.shape, 3), dtype=np.uint8)
    heat[..., 0] = scaled
    heat[..., 1] = np.clip(scaled.astype(np.int16) // 2, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(255 - scaled.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(heat)


def gray(arr: np.ndarray) -> np.ndarray:
    return (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114).astype(np.float32)


def edge_map(gray_arr: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray_arr, dtype=np.float32)
    gy = np.zeros_like(gray_arr, dtype=np.float32)
    gx[:, 1:] = np.abs(gray_arr[:, 1:] - gray_arr[:, :-1])
    gy[1:, :] = np.abs(gray_arr[1:, :] - gray_arr[:-1, :])
    return np.minimum(255.0, gx + gy)


def duplicate_line(line: Dict[str, Any], primitives: Sequence[Dict[str, Any]]) -> bool:
    angle = line_angle(line)
    mx = (line["x1"] + line["x2"]) / 2
    my = (line["y1"] + line["y2"]) / 2
    for primitive in primitives:
        if primitive.get("type") != "line":
            continue
        other_angle = line_angle(primitive)
        angle_diff = abs(math.atan2(math.sin(angle - other_angle), math.cos(angle - other_angle)))
        angle_diff = min(angle_diff, math.pi - angle_diff)
        omx = (primitive["x1"] + primitive["x2"]) / 2
        omy = (primitive["y1"] + primitive["y2"]) / 2
        if angle_diff < math.radians(5) and math.hypot(mx - omx, my - omy) < 6:
            return True
    return False


def same_line(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a["primitiveIndex"] == b["primitiveIndex"]


def clone_primitives(primitives: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return json.loads(json.dumps(primitives))


def nearest_other_endpoint_distance(a: Dict[str, Any], b: Dict[str, Any], endpoints: Sequence[Dict[str, Any]]) -> float:
    midpoint = {"x": (a["point"]["x"] + b["point"]["x"]) / 2, "y": (a["point"]["y"] + b["point"]["y"]) / 2}
    distances = []
    for endpoint in endpoints:
        if endpoint is a or endpoint is b:
            continue
        distances.append(distance(midpoint, endpoint["point"]))
    return min(128.0, min(distances) if distances else 128.0)


def ray_intersection(p: Dict[str, float], angle_a: float, q: Dict[str, float], angle_b: float) -> Dict[str, float] | None:
    ax = math.cos(angle_a)
    ay = math.sin(angle_a)
    bx = math.cos(angle_b)
    by = math.sin(angle_b)
    denom = ax * by - ay * bx
    if abs(denom) < 1e-6:
        return None
    qpx = q["x"] - p["x"]
    qpy = q["y"] - p["y"]
    t = (qpx * by - qpy * bx) / denom
    return {"x": p["x"] + t * ax, "y": p["y"] + t * ay}


def distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def direction(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.atan2(b["y"] - a["y"], b["x"] - a["x"])


def line_angle(line: Dict[str, Any]) -> float:
    return math.atan2(line["y2"] - line["y1"], line["x2"] - line["x1"])


def abs_angle_diff(a: float, b: float) -> float:
    diff = abs(math.atan2(math.sin(a - b), math.cos(a - b)))
    return min(diff, math.pi - diff)


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


if __name__ == "__main__":
    main()
