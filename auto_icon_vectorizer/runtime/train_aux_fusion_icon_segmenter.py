from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from generate_spectral_evidence_bank import alpha_heat_image, paste_panel, binary_mask_image
from trace_icon_component import (
    SIZE,
    alpha_flood_without_purple_mask,
    constant_color_alpha_map,
    estimate_smooth_color_background_scaled,
    purple_alpha_band,
    robust_unit_normalize,
    spectral_high_high_alpha_projection,
    threshold_spectral_channel,
    warm_high_intensity_alpha_band,
)


ROOT = Path(__file__).resolve().parent
TRUTH_RUN = ROOT / "truth-stress-eval" / "latest-run.json"
OUT = ROOT / "nn-seg-results"
FEATURE_CACHE = OUT / "feature-cache-v2.npz"
EXTRA_RUN = ROOT / "nn-training-corpus-v2" / "latest-run.json"
FOCUS_IDS = ["s02-002", "s01-002", "s01-008", "s02-033", "s01-031"]
STRIPE_IDS = ["s01-007", "s01-030", "s01-035", "s02-005", "s02-037"]


@dataclass
class Pack:
    ids: list[str]
    icons: list[str]
    backgrounds: list[str]
    source_paths: list[str]
    origins: list[str]
    features: np.ndarray
    masks: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--seed", type=int, default=447)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--no-extra", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    pack = load_or_build_pack(force=args.force_features, include_extra=not args.no_extra)
    train_idx, val_idx = split_indices(pack.ids)

    device = choose_device()
    model = TinyGatedUNet(in_main=4, in_aux=pack.features.shape[1] - 4, base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    train_ds = IconFeatureDataset(pack, train_idx, augment=True)
    val_ds = IconFeatureDataset(pack, val_idx, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best = {"epoch": 0, "val_iou": -1.0, "threshold": 0.5}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, mask in train_loader:
            features = features.to(device)
            mask = mask.to(device)
            logits, boundary_logits = model(features)
            loss = segmentation_loss(logits, boundary_logits, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, train_loader, device)
            val_metrics = evaluate(model, val_loader, device)
            threshold, val_iou = choose_threshold(val_metrics["probs"], val_metrics["targets"])
            train_iou = mean_iou_at_threshold(train_metrics["probs"], train_metrics["targets"], threshold)
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "threshold": threshold,
                "train_iou": train_iou,
                "val_iou": val_iou,
                "val_precision": mean_precision_at_threshold(val_metrics["probs"], val_metrics["targets"], threshold),
                "val_recall": mean_recall_at_threshold(val_metrics["probs"], val_metrics["targets"], threshold),
            }
            history.append(row)
            print(
                f"epoch {epoch:03d} loss={row['loss']:.4f} thr={threshold:.2f} "
                f"trainIoU={train_iou:.4f} valIoU={val_iou:.4f} "
                f"valP/R={row['val_precision']:.3f}/{row['val_recall']:.3f}",
                flush=True,
            )
            if val_iou > best["val_iou"]:
                best = {"epoch": epoch, "val_iou": val_iou, "threshold": threshold}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "threshold": threshold,
                        "features": feature_names(),
                        "base": args.base,
                        "feature_count": int(pack.features.shape[1]),
                        "epoch": epoch,
                        "val_iou": val_iou,
                    },
                    OUT / "best-gated-unet.pt",
                )

    (OUT / "training-history.json").write_text(json.dumps(history, indent=2) + "\n")
    checkpoint = torch.load(OUT / "best-gated-unet.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    threshold = float(checkpoint["threshold"])
    render_diagnostics(pack, model, threshold, train_idx, val_idx, device)
    print(json.dumps({"best": best, "checkpoint": str(OUT / "best-gated-unet.pt")}, indent=2))


def load_or_build_pack(force: bool, include_extra: bool) -> Pack:
    if FEATURE_CACHE.exists() and not force:
        data = np.load(FEATURE_CACHE, allow_pickle=True)
        return Pack(
            ids=list(data["ids"]),
            icons=list(data["icons"]),
            backgrounds=list(data["backgrounds"]),
            source_paths=list(data["source_paths"]),
            origins=list(data["origins"]),
            features=data["features"].astype(np.float32),
            masks=data["masks"].astype(np.float32),
        )

    reports_with_origin: list[tuple[str, dict]] = [
        ("truth-stress-eval", report) for report in json.loads(TRUTH_RUN.read_text())["reports"]
    ]
    if include_extra and EXTRA_RUN.exists():
        reports_with_origin.extend(
            ("nn-training-corpus-v2", report) for report in json.loads(EXTRA_RUN.read_text())["reports"]
        )
    ids: list[str] = []
    icons: list[str] = []
    backgrounds: list[str] = []
    source_paths: list[str] = []
    origins: list[str] = []
    features = []
    masks = []
    for origin, report in reports_with_origin:
        source = Image.open(report["sourceCrop"]).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        truth = truth_mask(report)
        feature = build_feature_stack(source)
        ids.append(report["id"])
        icons.append(report["icon"])
        backgrounds.append(report["backgroundMode"])
        source_paths.append(report["sourceCrop"])
        origins.append(origin)
        features.append(feature)
        masks.append(truth.astype(np.float32)[None, ...])
        print("features", origin, report["id"], report["icon"], feature.shape, flush=True)

    pack = Pack(
        ids=ids,
        icons=icons,
        backgrounds=backgrounds,
        source_paths=source_paths,
        origins=origins,
        features=np.stack(features).astype(np.float32),
        masks=np.stack(masks).astype(np.float32),
    )
    np.savez_compressed(
        FEATURE_CACHE,
        ids=np.asarray(pack.ids),
        icons=np.asarray(pack.icons),
        backgrounds=np.asarray(pack.backgrounds),
        source_paths=np.asarray(pack.source_paths),
        origins=np.asarray(pack.origins),
        features=pack.features,
        masks=pack.masks,
    )
    return pack


def build_feature_stack(source: Image.Image) -> np.ndarray:
    rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    rgb_u8 = (rgb * 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    alpha, _ = constant_color_alpha_map(source)
    warm = warm_high_intensity_alpha_band(alpha).astype(np.float32)
    purple = purple_alpha_band(alpha).astype(np.float32)
    joined = np.maximum(warm, purple)
    high_high, best_channel, _, _ = spectral_high_high_alpha_projection(source, alpha)
    high_bw = threshold_feature(high_high)
    best_bw = threshold_spectral_channel(best_channel).astype(np.float32)
    base = alpha_flood_without_purple_mask(alpha).astype(np.float32)
    sobel_x = cv2.Sobel(alpha.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(alpha.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    alpha_grad = np.clip(np.sqrt(np.square(sobel_x) + np.square(sobel_y)), 0.0, 1.0)
    bg_dark = local_contrast(gray, dark=True)
    bg_light = local_contrast(gray, dark=False)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_bg = estimate_smooth_color_background_scaled(lab, 31, 8.0)
    lab_residual = lab - lab_bg
    lab_residual_mag = robust_unit_normalize(np.linalg.norm(lab_residual, axis=2))
    ab_residual_mag = robust_unit_normalize(np.linalg.norm(lab_residual[..., 1:3], axis=2))
    rgb_bg = estimate_smooth_color_background_scaled(rgb * 255.0, 31, 8.0) / 255.0
    rgb_residual_mag = robust_unit_normalize(np.linalg.norm(rgb - rgb_bg, axis=2))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    xx = (xx / max(1, SIZE - 1)) * 2.0 - 1.0
    yy = (yy / max(1, SIZE - 1)) * 2.0 - 1.0

    channels = [
        rgb[..., 0],
        rgb[..., 1],
        rgb[..., 2],
        alpha,
        warm,
        purple,
        joined,
        high_high,
        high_bw,
        best_channel,
        best_bw,
        base,
        alpha_grad,
        bg_dark,
        bg_light,
        saturation,
        value,
        lab_residual_mag,
        ab_residual_mag,
        rgb_residual_mag,
        xx,
        yy,
    ]
    return np.stack(channels).astype(np.float32)


def threshold_feature(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0.02]
    if positive.size < 24:
        return np.zeros(values.shape, dtype=np.float32)
    try:
        from skimage.filters import threshold_otsu

        otsu = float(threshold_otsu(positive))
    except Exception:
        otsu = float(np.percentile(positive, 45))
    threshold = max(0.045, min(0.30, min(otsu, float(np.percentile(positive, 35)))))
    return (values >= threshold).astype(np.float32)


def local_contrast(gray: np.ndarray, dark: bool) -> np.ndarray:
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
    if dark:
        return np.clip(background - gray, 0.0, 1.0)
    return np.clip(gray - background, 0.0, 1.0)


class IconFeatureDataset(Dataset):
    def __init__(self, pack: Pack, indices: Sequence[int], augment: bool) -> None:
        self.pack = pack
        self.indices = list(indices)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        x = self.pack.features[index].copy()
        y = self.pack.masks[index].copy()
        if self.augment:
            x, y = augment_pair(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)


def augment_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        x = x[..., ::-1].copy()
        y = y[..., ::-1].copy()
        x[-2] = -x[-2]
    if random.random() < 0.5:
        x = x[..., ::-1, :].copy()
        y = y[..., ::-1, :].copy()
        x[-1] = -x[-1]
    k = random.randint(0, 3)
    if k:
        x = np.rot90(x, k=k, axes=(1, 2)).copy()
        y = np.rot90(y, k=k, axes=(1, 2)).copy()
    if random.random() < 0.35:
        # Modality dropout forces the model not to over-trust one auxiliary map.
        aux_channels = list(range(4, x.shape[0] - 2))
        random.shuffle(aux_channels)
        for channel in aux_channels[: random.randint(1, 3)]:
            x[channel] = 0.0
    if random.random() < 0.35:
        gain = random.uniform(0.85, 1.18)
        bias = random.uniform(-0.055, 0.055)
        x[:4] = np.clip(x[:4] * gain + bias, 0.0, 1.0)
    if random.random() < 0.25:
        x[:3] = np.clip(x[:3] + np.random.normal(0.0, 0.025, size=x[:3].shape), 0.0, 1.0)
    return x.astype(np.float32), y.astype(np.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(3 if out_ch % 3 == 0 else 1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(3 if out_ch % 3 == 0 else 1, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFuse(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())

    def forward(self, main: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([main, aux], dim=1))
        return main + gate * aux


class TinyGatedUNet(nn.Module):
    def __init__(self, in_main: int, in_aux: int, base: int) -> None:
        super().__init__()
        self.main1 = ConvBlock(in_main, base)
        self.aux1 = ConvBlock(in_aux, base)
        self.fuse1 = GatedFuse(base)
        self.main2 = ConvBlock(base, base * 2)
        self.aux2 = ConvBlock(base, base * 2)
        self.fuse2 = GatedFuse(base * 2)
        self.main3 = ConvBlock(base * 2, base * 4)
        self.aux3 = ConvBlock(base * 2, base * 4)
        self.fuse3 = GatedFuse(base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.mask_head = nn.Conv2d(base, 1, 1)
        self.boundary_head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        main = x[:, :4]
        aux = x[:, 4:]
        m1 = self.main1(main)
        a1 = self.aux1(aux)
        f1 = self.fuse1(m1, a1)
        m2 = self.main2(F.max_pool2d(f1, 2))
        a2 = self.aux2(F.max_pool2d(a1, 2))
        f2 = self.fuse2(m2, a2)
        m3 = self.main3(F.max_pool2d(f2, 2))
        a3 = self.aux3(F.max_pool2d(a2, 2))
        f3 = self.fuse3(m3, a3)
        d2 = self.up2(f3)
        d2 = self.dec2(torch.cat([d2, f2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, f1], dim=1))
        return self.mask_head(d1), self.boundary_head(d1)


def segmentation_loss(logits: torch.Tensor, boundary_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = weighted_bce(logits, target)
    prob = torch.sigmoid(logits)
    dice = dice_loss(prob, target)
    tversky = tversky_loss(prob, target)
    boundary_target = boundary_from_mask(target)
    boundary_bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
    return bce + 0.55 * dice + 0.45 * tversky + 0.24 * boundary_bce


def weighted_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Thin icon strokes make false negatives more damaging than small edge extras.
    weight = torch.where(target > 0.5, torch.full_like(target, 1.85), torch.ones_like(target))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def tversky_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    tp = (prob * target).sum(dim=dims)
    fp = (prob * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - prob) * target).sum(dim=dims)
    score = (tp + 1.0) / (tp + 0.38 * fp + 0.62 * fn + 1.0)
    return 1.0 - score.mean()


def boundary_from_mask(mask: torch.Tensor) -> torch.Tensor:
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    probs = []
    targets = []
    for features, mask in loader:
        features = features.to(device)
        logits, _ = model(features)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(mask.numpy())
    return {"probs": np.concatenate(probs, axis=0), "targets": np.concatenate(targets, axis=0)}


def choose_threshold(probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_iou = -1.0
    for threshold in np.linspace(0.18, 0.78, 31):
        score = mean_iou_at_threshold(probs, targets, float(threshold))
        if score > best_iou:
            best_iou = score
            best_t = float(threshold)
    return best_t, best_iou


def mean_iou_at_threshold(probs: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    truth = targets >= 0.5
    scores = []
    for p, t in zip(pred, truth):
        scores.append(float(np.logical_and(p, t).sum() / max(1, np.logical_or(p, t).sum())))
    return float(np.mean(scores))


def mean_precision_at_threshold(probs: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    truth = targets >= 0.5
    scores = []
    for p, t in zip(pred, truth):
        scores.append(float(np.logical_and(p, t).sum() / max(1, p.sum())))
    return float(np.mean(scores))


def mean_recall_at_threshold(probs: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    truth = targets >= 0.5
    scores = []
    for p, t in zip(pred, truth):
        scores.append(float(np.logical_and(p, t).sum() / max(1, t.sum())))
    return float(np.mean(scores))


@torch.no_grad()
def render_diagnostics(
    pack: Pack,
    model: nn.Module,
    threshold: float,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    device: str,
) -> None:
    all_probs = predict_all(pack, model, device)
    render_selection(pack, all_probs, threshold, familiar_indices(pack), OUT / "familiar-examples-gated-unet.png", "Tiny gated U-Net: familiar examples")
    worst_val = sorted(val_idx, key=lambda idx: single_iou(all_probs[idx], pack.masks[idx], threshold))[: min(16, len(val_idx))]
    render_selection(pack, all_probs, threshold, worst_val, OUT / "worst-validation-gated-unet.png", "Tiny gated U-Net: worst validation cases")
    write_prediction_summary(pack, all_probs, threshold, train_idx, val_idx, OUT / "prediction-summary.tsv")


def predict_all(pack: Pack, model: nn.Module, device: str) -> np.ndarray:
    model.eval()
    probs = []
    for start in range(0, len(pack.ids), 8):
        x = torch.from_numpy(pack.features[start : start + 8]).to(device)
        logits, _ = model(x)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def render_selection(pack: Pack, probs: np.ndarray, threshold: float, indices: Sequence[int], path: Path, title: str) -> None:
    panel = 108
    gap = 12
    left = 238
    header = 78
    row_h = panel + 66
    columns = ["source", "truth", "alpha", "joined bands", "base flood", "lab resid", "probability", "pred BW", "error"]
    canvas = Image.new("RGB", (left + len(columns) * panel + (len(columns) + 1) * gap, header + len(indices) * row_h + gap), "#f7f6f2")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), f"{title} threshold={threshold:.2f}", fill="#222", font=font)
    draw.text((12, 30), "Error: black=correct, red=extra, blue=missed.", fill="#555", font=font)
    x = left + gap
    for column in columns:
        draw.text((x, 54), column, fill="#222", font=font)
        x += panel + gap
    for row, idx in enumerate(indices):
        source = Image.open(pack.source_paths[idx]).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        truth = pack.masks[idx, 0]
        alpha = pack.features[idx, 3]
        joined = pack.features[idx, 6]
        base = pack.features[idx, 11]
        lab_resid = pack.features[idx, 17]
        prob = probs[idx, 0]
        pred = (prob >= threshold).astype(np.uint8)
        score = single_iou(prob, pack.masks[idx], threshold)
        precision = single_precision(prob, pack.masks[idx], threshold)
        recall = single_recall(prob, pack.masks[idx], threshold)
        y = header + row * row_h + gap
        draw.text((12, y + 2), f"{pack.ids[idx]} {pack.icons[idx]}", fill="#111", font=font)
        draw.text((12, y + 18), f"bg={pack.backgrounds[idx]} IoU={score:.3f}", fill="#444", font=font)
        draw.text((12, y + 34), f"P/R {precision:.2f}/{recall:.2f}", fill="#444", font=font)
        panels = [
            source,
            binary_mask_image(truth.astype(np.uint8)),
            alpha_heat_image(alpha),
            joined_image(joined),
            binary_mask_image(base.astype(np.uint8)),
            gray_feature_image(lab_resid),
            probability_image(prob),
            binary_mask_image(pred),
            error_image(pred, truth),
        ]
        x = left + gap
        for image in panels:
            paste_panel(canvas, draw, image, x, y, panel)
            x += panel + gap
    canvas.save(path)
    print(path)


def familiar_indices(pack: Pack) -> list[int]:
    wanted = FOCUS_IDS + STRIPE_IDS
    lookup = {ident: idx for idx, ident in enumerate(pack.ids)}
    return [lookup[ident] for ident in wanted if ident in lookup]


def write_prediction_summary(
    pack: Pack,
    probs: np.ndarray,
    threshold: float,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    path: Path,
) -> None:
    train_set = set(train_idx)
    val_set = set(val_idx)
    lines = ["split\tid\ticon\tbg\tiou\tprecision\trecall\tpred_px\ttruth_px"]
    for idx in range(len(pack.ids)):
        split = "train" if idx in train_set else "val" if idx in val_set else "other"
        prob = probs[idx]
        truth = pack.masks[idx]
        pred = prob >= threshold
        lines.append(
            f"{split}\t{pack.ids[idx]}\t{pack.icons[idx]}\t{pack.backgrounds[idx]}\t"
            f"{single_iou(prob, truth, threshold):.4f}\t{single_precision(prob, truth, threshold):.4f}\t"
            f"{single_recall(prob, truth, threshold):.4f}\t{int(pred.sum())}\t{int((truth >= 0.5).sum())}"
        )
    path.write_text("\n".join(lines) + "\n")
    print(path)


def single_iou(prob: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    pred = prob >= threshold
    target = truth >= 0.5
    return float(np.logical_and(pred, target).sum() / max(1, np.logical_or(pred, target).sum()))


def single_precision(prob: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    pred = prob >= threshold
    target = truth >= 0.5
    return float(np.logical_and(pred, target).sum() / max(1, pred.sum()))


def single_recall(prob: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    pred = prob >= threshold
    target = truth >= 0.5
    return float(np.logical_and(pred, target).sum() / max(1, target.sum()))


def joined_image(joined: np.ndarray) -> Image.Image:
    image = np.zeros((joined.shape[0], joined.shape[1], 3), dtype=np.uint8)
    image[joined.astype(bool)] = (235, 135, 18)
    return Image.fromarray(image, "RGB")


def probability_image(prob: np.ndarray) -> Image.Image:
    prob = np.clip(prob, 0.0, 1.0)
    bgr = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "RGB")


def gray_feature_image(values: np.ndarray) -> Image.Image:
    g = (255 * (1.0 - np.clip(values, 0.0, 1.0))).astype(np.uint8)
    return Image.fromarray(np.dstack([g, g, g]), "RGB")


def error_image(mask: np.ndarray, truth: np.ndarray) -> Image.Image:
    pred = mask.astype(bool)
    target = truth.astype(bool)
    image = np.ones((truth.shape[0], truth.shape[1], 3), dtype=np.uint8) * 250
    image[pred & target] = (0, 0, 0)
    image[pred & ~target] = (220, 35, 35)
    image[~pred & target] = (45, 100, 220)
    return Image.fromarray(image, "RGB")


def split_indices(ids: Sequence[str]) -> tuple[list[int], list[int]]:
    original_count = len(json.loads(TRUTH_RUN.read_text())["reports"])
    original_ids = list(ids[:original_count])
    val = [
        idx
        for idx, ident in enumerate(original_ids)
        if ident.startswith("s") and int(ident[1:3]) == 2 and int(ident[-3:]) % 4 == 1
    ]
    if len(val) < 12:
        val = [idx for idx in range(original_count) if idx % 5 == 0]
    train = [idx for idx in range(len(ids)) if idx not in set(val)]
    return train, val


def truth_mask(report: dict) -> np.ndarray:
    return (np.asarray(Image.open(report["truthIcon"]).convert("RGBA"))[..., 3] > 20).astype(np.float32)


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_names() -> list[str]:
    return [
        "rgb_r",
        "rgb_g",
        "rgb_b",
        "alpha",
        "warm",
        "purple",
        "joined",
        "high_high",
        "high_bw",
        "best_channel",
        "best_bw",
        "base_flood",
        "alpha_grad",
        "gray_dark",
        "gray_light",
        "saturation",
        "value",
        "lab_residual_mag",
        "ab_residual_mag",
        "rgb_residual_mag",
        "x",
        "y",
    ]


if __name__ == "__main__":
    main()
