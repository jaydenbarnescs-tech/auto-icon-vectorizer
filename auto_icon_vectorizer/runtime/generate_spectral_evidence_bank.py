from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_otsu

from trace_icon_component import (
    SIZE,
    clean_mask,
    constant_color_alpha_map,
    estimate_smooth_color_background_scaled,
    fill_tiny_white_slits,
    purple_alpha_band,
    robust_unit_normalize,
    seeded_connected_components,
    suppress_background_texture_components,
    warm_high_intensity_alpha_band,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "spectral-evidence-bank-results"
TRUTH_RUN = ROOT / "truth-stress-eval" / "latest-run.json"
ORANGE = (255, 151, 82)
PURPLE = (90, 49, 180)


@dataclass
class SpectralCandidate:
    name: str
    angle: float
    channel: np.ndarray
    color: tuple[int, int, int]
    score: float
    signed_fit: float
    warm_fit: float
    purple_fit: float
    warm_ratio: float
    purple_ratio: float
    border_penalty: float
    area: float


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = json.loads(TRUTH_RUN.read_text())["reports"]
    by_id = {report["id"]: report for report in reports}
    ids = ["s02-006", "s01-015", "s02-037", "s02-033", "s02-040"]

    make_detail_sheet(by_id["s02-037"], OUT / "s02-037-spectrum-bank-detail.png")
    make_summary_sheet([by_id[ident] for ident in ids], OUT / "spectral-bank-five-example-summary.png")


def make_detail_sheet(report: dict, path: Path) -> None:
    source = load_source(report)
    alpha, _ = constant_color_alpha_map(source)
    candidates = sorted(generate_spectral_candidates(source, alpha), key=lambda item: item.score, reverse=True)
    fused = fuse_candidates(candidates[:5])
    fused_mask = mask_from_fused_channel(fused, alpha)

    panel = 112
    gap = 12
    left = 190
    header = 148
    cols = 6
    rows = math.ceil(len(candidates) / cols)
    width = left + cols * panel + (cols + 1) * gap
    height = header + rows * (panel + 38) + gap
    canvas = Image.new("RGB", (width, height), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 12), f"{report['id']} {report['icon']} - spectrum channel bank", fill="#222", font=font)
    draw.text((12, 30), "score = warm/yellow alpha agreement - purple alpha agreement - stripe/border penalties", fill="#444", font=font)

    overview = [
        ("source", source),
        ("raw alpha heat", alpha_heat_image(alpha)),
        ("warm positive", band_image(alpha, warm_high_intensity_alpha_band(alpha), ORANGE)),
        ("purple negative", band_image(alpha, purple_alpha_band(alpha), PURPLE)),
        ("top5 fused", channel_image(fused, (220, 220, 220))),
        ("fused mask", binary_mask_image(fused_mask)),
    ]
    x = left + gap
    for label, image in overview:
        draw.text((x, 56), label, fill="#222", font=font)
        paste_panel(canvas, draw, image, x, 72, panel)
        x += panel + gap

    top_names = {candidate.name for candidate in candidates[:5]}
    y0 = header
    for index, candidate in enumerate(candidates):
        col = index % cols
        row = index // cols
        x = left + gap + col * (panel + gap)
        y = y0 + row * (panel + 38)
        outline = "#238636" if candidate.name in top_names else "#b9b5aa"
        if candidate.purple_ratio > candidate.warm_ratio and candidate.purple_fit > 0.18:
            outline = "#7a3fb0"
        draw.rectangle((x - 2, y - 2, x + panel + 1, y + panel + 1), outline=outline, width=2)
        canvas.paste(channel_image(candidate.channel, candidate.color).resize((panel, panel), Image.Resampling.NEAREST), (x, y))
        draw.text((x, y + panel + 4), f"{candidate.name} score {candidate.score:.2f}", fill="#222", font=font)
        draw.text(
            (x, y + panel + 18),
            f"+{candidate.warm_fit:.2f} -{candidate.purple_fit:.2f} b{candidate.border_penalty:.2f}",
            fill="#555",
            font=font,
        )

    draw.text((12, 72), "Top channels are green.", fill="#444", font=font)
    draw.text((12, 88), "Purple outline means rejected by purple/stripe evidence.", fill="#444", font=font)
    draw.text((12, 104), f"Best: {candidates[0].name} score {candidates[0].score:.2f}", fill="#111", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_summary_sheet(reports: Sequence[dict], path: Path) -> None:
    panel = 128
    gap = 14
    left = 198
    header = 38
    row_h = panel + 54
    columns = ["source", "raw alpha heat", "warm +", "purple -", "best channel", "top5 fused", "fused mask"]
    width = left + len(columns) * panel + (len(columns) + 1) * gap
    height = header + len(reports) * row_h + gap
    canvas = Image.new("RGB", (width, height), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((12, 12), "spectral evidence bank: alpha heat rewards warm/yellow and penalizes purple", fill="#222", font=font)
    x = left + gap
    for label in columns:
        draw.text((x, 14), label, fill="#222", font=font)
        x += panel + gap

    rows = ["id\ticon\tbg\tbest\tscore\twarm_fit\tpurple_fit\tborder_penalty\tmask_iou"]
    for row_index, report in enumerate(reports):
        source = load_source(report)
        alpha, _ = constant_color_alpha_map(source)
        candidates = sorted(generate_spectral_candidates(source, alpha), key=lambda item: item.score, reverse=True)
        best = candidates[0]
        fused = fuse_candidates(candidates[:5])
        fused_mask = mask_from_fused_channel(fused, alpha)
        truth = truth_mask(report)
        score_iou = mask_iou(fused_mask, truth)
        rows.append(
            f"{report['id']}\t{report['icon']}\t{report['backgroundMode']}\t{best.name}\t"
            f"{best.score:.4f}\t{best.warm_fit:.4f}\t{best.purple_fit:.4f}\t{best.border_penalty:.4f}\t{score_iou:.4f}"
        )

        y = header + row_index * row_h + gap
        draw.text((12, y + 2), f"{report['id']} {report['icon']}", fill="#111", font=font)
        draw.text((12, y + 18), f"bg={report['backgroundMode']}", fill="#444", font=font)
        draw.text((12, y + 34), f"best {best.name} score {best.score:.2f}", fill="#444", font=font)
        draw.text((12, y + 50), f"+{best.warm_fit:.2f} -{best.purple_fit:.2f} IoU {score_iou:.3f}", fill="#555", font=font)

        panels = [
            source,
            alpha_heat_image(alpha),
            band_image(alpha, warm_high_intensity_alpha_band(alpha), ORANGE),
            band_image(alpha, purple_alpha_band(alpha), PURPLE),
            channel_image(best.channel, best.color),
            channel_image(fused, (220, 220, 220)),
            binary_mask_image(fused_mask),
        ]
        x = left + gap
        for image in panels:
            paste_panel(canvas, draw, image, x, y, panel)
            x += panel + gap

    canvas.save(path)
    path.with_suffix(".tsv").write_text("\n".join(rows) + "\n")


def generate_spectral_candidates(source: Image.Image, alpha: np.ndarray) -> list[SpectralCandidate]:
    rgb_u8 = np.asarray(source.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_bg = estimate_smooth_color_background_scaled(lab, 31, 8.0)
    lab_residual = lab - lab_bg
    ab = lab_residual[..., 1:3]

    candidates: list[SpectralCandidate] = []
    for index, angle in enumerate(np.linspace(0, 2 * math.pi, 24, endpoint=False)):
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        raw = np.maximum(0.0, ab[..., 0] * direction[0] + ab[..., 1] * direction[1])
        channel = robust_unit_normalize(raw)
        color = hsv_rgb(angle)
        candidates.append(score_candidate(f"lab{index:02d}", float(math.degrees(angle)), channel, color, alpha))

    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_bg = cv2.GaussianBlur(cv2.medianBlur(gray.astype(np.uint8), 31).astype(np.float32), (0, 0), 8.0)
    dark = robust_unit_normalize(np.maximum(gray_bg - gray, 0.0))
    light = robust_unit_normalize(np.maximum(gray - gray_bg, 0.0))
    candidates.append(score_candidate("gray-dark", -1.0, dark, (40, 40, 40), alpha))
    candidates.append(score_candidate("gray-light", -1.0, light, (230, 230, 230), alpha))
    return candidates


def score_candidate(name: str, angle: float, channel: np.ndarray, color: tuple[int, int, int], alpha: np.ndarray) -> SpectralCandidate:
    channel = np.clip(channel.astype(np.float32), 0.0, 1.0)
    warm = warm_high_intensity_alpha_band(alpha)
    purple = purple_alpha_band(alpha)
    warm_weight = np.where(warm, np.clip(alpha, 0.0, 1.0) ** 1.35, 0.0)
    purple_weight = np.where(purple, np.clip(alpha, 0.0, 1.0) ** 1.35, 0.0)
    warm_fit = weighted_mean(channel, warm_weight)
    purple_fit = weighted_mean(channel, purple_weight)
    signed_fit = weighted_mean(channel, warm_weight - purple_weight)
    energy = float(channel.sum()) + 1e-6
    warm_ratio = float((channel * warm.astype(np.float32)).sum() / energy)
    purple_ratio = float((channel * purple.astype(np.float32)).sum() / energy)
    mask = threshold_channel(channel)
    border_penalty = border_texture_penalty(mask)
    area = float(mask.mean())
    area_penalty = abs(area - 0.12)
    score = (
        signed_fit * 2.2
        + warm_fit * 1.4
        + warm_ratio * 0.75
        - purple_fit * 1.7
        - purple_ratio * 1.6
        - border_penalty * 1.1
        - area_penalty * 0.35
    )
    return SpectralCandidate(
        name=name,
        angle=angle,
        channel=channel,
        color=color,
        score=float(score),
        signed_fit=float(signed_fit),
        warm_fit=float(warm_fit),
        purple_fit=float(purple_fit),
        warm_ratio=float(warm_ratio),
        purple_ratio=float(purple_ratio),
        border_penalty=float(border_penalty),
        area=area,
    )


def threshold_channel(channel: np.ndarray) -> np.ndarray:
    positive = channel[channel > 0.02]
    if positive.size < 24:
        return np.zeros(channel.shape, dtype=np.uint8)
    try:
        threshold = float(threshold_otsu(positive))
    except Exception:
        threshold = float(np.percentile(positive, 72))
    threshold = max(0.18, min(0.82, threshold))
    return (channel >= threshold).astype(np.uint8)


def border_texture_penalty(mask: np.ndarray) -> float:
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


def fuse_candidates(candidates: Sequence[SpectralCandidate]) -> np.ndarray:
    if not candidates:
        raise ValueError("no candidates to fuse")
    min_score = min(candidate.score for candidate in candidates)
    weights = np.array([max(0.05, candidate.score - min_score + 0.05) for candidate in candidates], dtype=np.float32)
    fused = np.zeros_like(candidates[0].channel, dtype=np.float32)
    for weight, candidate in zip(weights, candidates):
        fused += float(weight) * candidate.channel
    fused /= max(1e-6, float(weights.sum()))
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def mask_from_fused_channel(fused: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    warm = warm_high_intensity_alpha_band(alpha)
    positive = fused[fused > 0.02]
    if positive.size < 24:
        return np.zeros(fused.shape, dtype=np.uint8)
    seed = (fused >= max(0.38, float(np.percentile(positive, 78)))) & warm
    if int(seed.sum()) < 16:
        seed = fused >= max(0.38, float(np.percentile(positive, 86)))
    support = fused >= max(0.08, float(np.percentile(positive, 35)))
    mask = seeded_connected_components(support, seed)
    mask = clean_mask(mask)
    mask = fill_tiny_white_slits(mask, max_area=24, max_span=9)
    return suppress_background_texture_components(mask).astype(np.uint8)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.abs(weights).sum())
    if denom <= 1e-6:
        return 0.0
    return float((values * weights).sum() / denom)


def hsv_rgb(angle: float) -> tuple[int, int, int]:
    hue = int((angle % (2 * math.pi)) / (2 * math.pi) * 179)
    bgr = cv2.cvtColor(np.array([[[hue, 210, 240]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def load_source(report: dict) -> Image.Image:
    return Image.open(report["sourceCrop"]).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)


def truth_mask(report: dict) -> np.ndarray:
    return (np.asarray(Image.open(report["truthIcon"]).convert("RGBA"))[..., 3] > 20).astype(np.uint8)


def mask_iou(mask: np.ndarray, truth: np.ndarray) -> float:
    p = mask.astype(bool)
    t = truth.astype(bool)
    return float(np.logical_and(p, t).sum() / max(1, np.logical_or(p, t).sum()))


def alpha_heat_image(alpha: np.ndarray) -> Image.Image:
    bgr = cv2.applyColorMap((np.clip(alpha, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "RGB")


def band_image(alpha: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    image = np.zeros((alpha.shape[0], alpha.shape[1], 3), dtype=np.uint8)
    strength = np.clip(alpha, 0, 1)
    for channel_index, value in enumerate(color):
        image[..., channel_index] = np.where(mask, strength * value, 0).astype(np.uint8)
    return Image.fromarray(image, "RGB")


def channel_image(channel: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    values = np.clip(channel, 0, 1)
    image = np.zeros((values.shape[0], values.shape[1], 3), dtype=np.uint8)
    for channel_index, value in enumerate(color):
        image[..., channel_index] = np.clip(values * value, 0, 255).astype(np.uint8)
    return Image.fromarray(image, "RGB")


def binary_mask_image(mask: np.ndarray) -> Image.Image:
    value = (255 * (1 - mask.astype(np.uint8))).astype(np.uint8)
    return Image.fromarray(np.dstack([value, value, value]), "RGB")


def paste_panel(canvas: Image.Image, draw: ImageDraw.ImageDraw, image: Image.Image, x: int, y: int, size: int) -> None:
    draw.rectangle((x - 1, y - 1, x + size, y + size), outline="#d0cdc4")
    canvas.paste(image.resize((size, size), Image.Resampling.NEAREST), (x, y))


if __name__ == "__main__":
    main()
