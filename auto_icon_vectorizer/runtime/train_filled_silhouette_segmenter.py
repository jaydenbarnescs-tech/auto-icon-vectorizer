from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset

from train_aux_fusion_icon_segmenter import TinyGatedUNet, choose_device


ROOT = Path(__file__).resolve().parent
SIZE = 128
OUT = ROOT / "nn-seg-results"
CHECKPOINT = OUT / "best-filled-silhouette-unet.pt"
FEATURE_CACHE = OUT / "filled-silhouette-feature-cache-v1.npz"


@dataclass
class Pack:
    ids: list[str]
    backgrounds: list[str]
    shapes: list[str]
    features: np.ndarray
    masks: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=900)
    parser.add_argument("--val", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--base", type=int, default=20)
    parser.add_argument("--seed", type=int, default=928)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    pack = load_or_build_pack(args.train, args.val, args.seed, args.force_features)
    train_idx = list(range(args.train))
    val_idx = list(range(args.train, args.train + args.val))

    device = choose_device()
    model = TinyGatedUNet(in_main=4, in_aux=pack.features.shape[1] - 4, base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    train_loader = DataLoader(FilledFeatureDataset(pack, train_idx, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FilledFeatureDataset(pack, val_idx, augment=False), batch_size=args.batch_size, shuffle=False)

    best = {"epoch": 0, "val_iou": -1.0, "threshold": 0.5}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, mask in train_loader:
            features = features.to(device)
            mask = mask.to(device)
            logits, boundary_logits = model(features)
            loss = silhouette_loss(logits, boundary_logits, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader, device)
            threshold, val_iou = choose_threshold(val["probs"], val["targets"])
            precision = mean_precision_at_threshold(val["probs"], val["targets"], threshold)
            recall = mean_recall_at_threshold(val["probs"], val["targets"], threshold)
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "threshold": threshold,
                "val_iou": val_iou,
                "val_precision": precision,
                "val_recall": recall,
            }
            history.append(row)
            print(
                f"epoch {epoch:03d} loss={row['loss']:.4f} thr={threshold:.2f} "
                f"valIoU={val_iou:.4f} valP/R={precision:.3f}/{recall:.3f}",
                flush=True,
            )
            if val_iou > best["val_iou"]:
                best = {"epoch": epoch, "val_iou": val_iou, "threshold": threshold}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "threshold": threshold,
                        "base": args.base,
                        "feature_count": int(pack.features.shape[1]),
                        "epoch": epoch,
                        "val_iou": val_iou,
                    },
                    CHECKPOINT,
                )

    (OUT / "filled-silhouette-training-history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"best": best, "checkpoint": str(CHECKPOINT)}, indent=2))


def load_or_build_pack(train_count: int, val_count: int, seed: int, force: bool) -> Pack:
    expected = train_count + val_count
    if FEATURE_CACHE.exists() and not force:
        data = np.load(FEATURE_CACHE, allow_pickle=True)
        if int(data["features"].shape[0]) == expected:
            return Pack(
                ids=list(data["ids"]),
                backgrounds=list(data["backgrounds"]),
                shapes=list(data["shapes"]),
                features=data["features"].astype(np.float32),
                masks=data["masks"].astype(np.float32),
            )

    features = []
    masks = []
    ids = []
    backgrounds = []
    shapes = []
    for index in range(expected):
        source, truth, meta = generate_filled_icon(seed + index * 17)
        features.append(build_filled_feature_stack(source))
        masks.append(truth.astype(np.float32)[None, ...])
        ids.append(f"filled_{index:04d}")
        backgrounds.append(meta["background"])
        shapes.append(meta["shape"])
        if index % 50 == 0:
            print("features", index, meta, flush=True)

    pack = Pack(
        ids=ids,
        backgrounds=backgrounds,
        shapes=shapes,
        features=np.stack(features).astype(np.float32),
        masks=np.stack(masks).astype(np.float32),
    )
    np.savez_compressed(
        FEATURE_CACHE,
        ids=np.asarray(pack.ids),
        backgrounds=np.asarray(pack.backgrounds),
        shapes=np.asarray(pack.shapes),
        features=pack.features,
        masks=pack.masks,
    )
    return pack


class FilledFeatureDataset(Dataset):
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


def build_filled_feature_stack(source: Image.Image) -> np.ndarray:
    image = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    rgb_u8 = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0] / 179.0
    hue_sin = np.sin(hue * np.pi * 2.0).astype(np.float32)
    hue_cos = np.cos(hue * np.pi * 2.0).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_bg = smooth_background(lab, 35, 9.0)
    rgb_bg = smooth_background(rgb * 255.0, 35, 9.0) / 255.0
    lab_residual = normalize_unit(np.linalg.norm(lab - lab_bg, axis=2))
    ab_residual = normalize_unit(np.linalg.norm((lab - lab_bg)[..., 1:3], axis=2))
    rgb_residual = normalize_unit(np.linalg.norm(rgb - rgb_bg, axis=2))
    border_lab = np.median(border_pixels(lab), axis=0)
    border_rgb = np.median(border_pixels(rgb), axis=0)
    lab_border_distance = normalize_unit(np.linalg.norm(lab - border_lab[None, None, :], axis=2))
    rgb_border_distance = normalize_unit(np.linalg.norm(rgb - border_rgb[None, None, :], axis=2))
    bg_gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 5.0)
    local_dark = np.clip(bg_gray - gray, 0.0, 1.0)
    local_light = np.clip(gray - bg_gray, 0.0, 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = normalize_unit(np.sqrt(gx * gx + gy * gy))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    xx = (xx / max(1, SIZE - 1)) * 2.0 - 1.0
    yy = (yy / max(1, SIZE - 1)) * 2.0 - 1.0
    center_distance = np.clip(np.sqrt(xx * xx + yy * yy) / np.sqrt(2.0), 0.0, 1.0)
    return np.stack(
        [
            rgb[..., 0],
            rgb[..., 1],
            rgb[..., 2],
            gray,
            hue_sin,
            hue_cos,
            saturation,
            value,
            lab_residual,
            ab_residual,
            rgb_residual,
            lab_border_distance,
            rgb_border_distance,
            local_dark,
            local_light,
            edge,
            center_distance,
            xx,
            yy,
        ]
    ).astype(np.float32)


def filled_silhouette_unet_mask(image: Image.Image) -> np.ndarray | None:
    if not CHECKPOINT.exists():
        return None
    try:
        device = choose_device()
        checkpoint = torch.load(CHECKPOINT, map_location=device)
        feature = build_filled_feature_stack(image).astype(np.float32)
        model = _get_model(device, checkpoint, feature.shape[0])
        with torch.no_grad():
            x = torch.from_numpy(feature[None, ...]).to(device)
            logits, _ = model(x)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        threshold = float(checkpoint.get("threshold", 0.5))
        mask = hysteresis_probability_mask(prob, threshold)
        return clean_silhouette_mask(mask.astype(np.uint8))
    except Exception:
        return None


_MODEL_CACHE: dict[str, object] | None = None


def _get_model(device: str, checkpoint: dict, feature_count: int) -> nn.Module:
    global _MODEL_CACHE
    if _MODEL_CACHE and _MODEL_CACHE["device"] == device and _MODEL_CACHE["feature_count"] == feature_count:
        return _MODEL_CACHE["model"]  # type: ignore[return-value]
    base = int(checkpoint.get("base", 20))
    expected_features = int(checkpoint.get("feature_count", feature_count))
    model = TinyGatedUNet(in_main=4, in_aux=expected_features - 4, base=base).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _MODEL_CACHE = {"device": device, "feature_count": expected_features, "model": model}
    return model


def clean_silhouette_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.astype(np.uint8)
    out = cv2.medianBlur(out * 255, 3) > 0
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    out = remove_small_components(out, min_area=48)
    # Preserve intentional cutouts such as tag holes and map-pin centers. Only
    # fill tiny raster pinholes that are too small to be an icon feature.
    out = fill_holes(out, max_area=24)
    return out.astype(np.uint8)


def hysteresis_probability_mask(prob: np.ndarray, threshold: float) -> np.ndarray:
    seed = prob >= threshold
    support = prob >= max(0.22, threshold - 0.18)
    count, labels, _, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
    out = np.zeros(prob.shape, dtype=np.uint8)
    for label in range(1, count):
        component = labels == label
        if np.logical_and(component, seed).any():
            out[component] = 1
    return out


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out


def fill_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
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


def silhouette_loss(logits: torch.Tensor, boundary_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(prob, target)
    tversky = tversky_loss(prob, target)
    boundary = boundary_from_mask(target)
    boundary_bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary)
    return bce + 0.80 * dice + 0.35 * tversky + 0.16 * boundary_bce


def dice_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    intersection = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0))).mean()


def tversky_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = (1, 2, 3)
    tp = (prob * target).sum(dim=dims)
    fp = (prob * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - prob) * target).sum(dim=dims)
    score = (tp + 1.0) / (tp + 0.44 * fp + 0.56 * fn + 1.0)
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
        logits, _ = model(features.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(mask.numpy())
    return {"probs": np.concatenate(probs, axis=0), "targets": np.concatenate(targets, axis=0)}


def choose_threshold(probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.linspace(0.28, 0.72, 23):
        iou = mean_iou_at_threshold(probs, targets, float(threshold))
        if iou > best[1]:
            best = (float(threshold), iou)
    return best


def mean_iou_at_threshold(probs: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    target = targets >= 0.5
    scores = []
    for p, t in zip(pred, target):
        intersection = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        scores.append(float(intersection / max(1, union)))
    return float(np.mean(scores))


def mean_precision_at_threshold(probs: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    target = targets >= 0.5
    tp = np.logical_and(pred, target).sum(axis=(1, 2, 3))
    fp = np.logical_and(pred, ~target).sum(axis=(1, 2, 3))
    return float(np.mean(tp / np.maximum(1, tp + fp)))


def mean_recall_at_threshold(probs: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    target = targets >= 0.5
    tp = np.logical_and(pred, target).sum(axis=(1, 2, 3))
    fn = np.logical_and(~pred, target).sum(axis=(1, 2, 3))
    return float(np.mean(tp / np.maximum(1, tp + fn)))


def augment_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        x = x[..., ::-1].copy()
        y = y[..., ::-1].copy()
        x[-2] = -x[-2]
    if random.random() < 0.5:
        x = x[..., ::-1, :].copy()
        y = y[..., ::-1, :].copy()
        x[-1] = -x[-1]
    if random.random() < 0.25:
        k = random.randint(0, 3)
        x = np.rot90(x, k=k, axes=(1, 2)).copy()
        y = np.rot90(y, k=k, axes=(1, 2)).copy()
    if random.random() < 0.35:
        gain = random.uniform(0.86, 1.16)
        bias = random.uniform(-0.045, 0.045)
        x[:4] = np.clip(x[:4] * gain + bias, 0.0, 1.0)
    if random.random() < 0.22:
        x[:3] = np.clip(x[:3] + np.random.normal(0.0, 0.022, size=x[:3].shape), 0.0, 1.0)
    return x.astype(np.float32), y.astype(np.float32)


ShapeDrawer = Callable[[Image.Image, tuple[int, int, int, int], random.Random], None]


def generate_filled_icon(seed: int) -> tuple[Image.Image, np.ndarray, dict[str, str]]:
    rng = random.Random(seed)
    scale = 4
    size = SIZE * scale
    background_mode = rng.choices(
        ["dark", "paper", "stripe", "color", "map", "soft_gradient", "split_map"],
        weights=[1.0, 1.0, 1.0, 1.0, 1.15, 1.0, 2.2],
        k=1,
    )[0]
    source = synthetic_background(size, background_mode, rng)
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    truth_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if background_mode == "split_map":
        shape_name, drawer = rng.choices(SHAPES, weights=[5.5, 0.4, 0.4, 0.8, 1.4, 0.4, 0.4, 0.4], k=1)[0]
    else:
        shape_name, drawer = rng.choice(SHAPES)
    color_choices = [
        (197, 72, 41, 255),
        (203, 168, 93, 255),
        (52, 50, 46, 255),
        (57, 96, 177, 255),
        (57, 135, 90, 255),
    ]
    if background_mode in {"dark", "stripe", "map"}:
        color_choices.append((238, 229, 207, 255))
    color = rng.choice(color_choices)
    state = rng.getstate()
    drawer(layer, color, rng)
    rng.setstate(state)
    drawer(truth_layer, color, rng)
    angle = rng.uniform(-13, 13)
    layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    truth_layer = truth_layer.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    if rng.random() < 0.8:
        blur = rng.uniform(0.15, 0.75)
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
        truth_layer = truth_layer.filter(ImageFilter.GaussianBlur(blur))
    source.alpha_composite(layer)
    source = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    truth = Image.fromarray((np.asarray(truth_layer)[..., 3] > 20).astype(np.uint8) * 255)
    truth = truth.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    truth_mask = (np.asarray(truth) > 30).astype(np.float32)
    return source, truth_mask, {"shape": shape_name, "background": background_mode}


def synthetic_background(size: int, mode: str, rng: random.Random) -> Image.Image:
    if mode in {"dark", "stripe"}:
        base = np.array((20, 23, 18), dtype=np.float32)
    elif mode == "paper":
        base = np.array((246, 243, 235), dtype=np.float32)
    elif mode == "map":
        base = np.array((82, 92, 64), dtype=np.float32)
    else:
        base = np.array((70, 72, 54), dtype=np.float32)
    image = Image.new("RGB", (size, size), tuple(base.astype(int)))
    draw = ImageDraw.Draw(image)
    if mode == "stripe":
        for x in range(-size, size * 2, rng.randint(16, 30)):
            draw.polygon([(x, 0), (x + 9, 0), (x + size + 9, size), (x + size, size)], fill=(37, 41, 58))
            draw.polygon([(x + 10, 0), (x + 20, 0), (x + size + 20, size), (x + size + 10, size)], fill=(25, 29, 24))
        image = image.filter(ImageFilter.GaussianBlur(0.4))
    elif mode == "color":
        for y in range(size):
            t = y / max(1, size - 1)
            c = np.array((92, 65, 50)) * (1 - t) + np.array((42, 82, 66)) * t
            draw.line([(0, y), (size, y)], fill=tuple(c.astype(int)))
        for _ in range(45):
            x = rng.randrange(size)
            y = rng.randrange(size)
            r = rng.randrange(8, 40)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=rng.choice([(110, 65, 125), (37, 83, 98), (145, 103, 42), (62, 64, 38)]))
        image = image.filter(ImageFilter.GaussianBlur(1.6))
    elif mode in {"map", "split_map"}:
        for _ in range(18):
            pts = [(rng.randrange(size), rng.randrange(size)) for _ in range(rng.randint(3, 6))]
            draw.line(pts, fill=rng.choice([(185, 172, 126), (55, 76, 61), (104, 112, 79)]), width=rng.randint(3, 8), joint="curve")
        for _ in range(18):
            x = rng.randrange(size)
            y = rng.randrange(size)
            r = rng.randrange(12, 34)
            draw.rectangle((x - r, y - r, x + r, y + r), fill=rng.choice([(68, 83, 53), (120, 102, 63), (45, 68, 58)]))
        image = image.filter(ImageFilter.GaussianBlur(1.2))
        if mode == "split_map":
            overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            odraw = ImageDraw.Draw(overlay)
            paper = rng.choice([(243, 238, 226, 255), (236, 225, 207, 255), (246, 242, 232, 255)])
            if rng.random() < 0.55:
                cut = rng.randint(int(size * 0.32), int(size * 0.58))
                odraw.rectangle((0, 0, cut, size), fill=paper)
            else:
                odraw.polygon(
                    [
                        (0, 0),
                        (rng.randint(int(size * 0.28), int(size * 0.56)), 0),
                        (rng.randint(int(size * 0.42), int(size * 0.72)), size),
                        (0, size),
                    ],
                    fill=paper,
                )
            if rng.random() < 0.6:
                odraw.line(
                    [(0, rng.randint(0, size)), (size, rng.randint(0, size))],
                    fill=(205, 193, 164, 255),
                    width=rng.randint(2, 7),
                )
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    elif mode == "soft_gradient":
        c1 = np.array(rng.choice([(25, 28, 24), (245, 239, 226), (75, 60, 44)]), dtype=np.float32)
        c2 = np.array(rng.choice([(94, 64, 42), (47, 86, 72), (39, 42, 62)]), dtype=np.float32)
        for y in range(size):
            t = y / max(1, size - 1)
            c = c1 * (1 - t) + c2 * t
            draw.line([(0, y), (size, y)], fill=tuple(c.astype(int)))
        image = image.filter(ImageFilter.GaussianBlur(1.2))
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(3.0, 10.0), (size, size, 3))
    arr = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def icon_box(rng: random.Random, size: int) -> tuple[float, float, float, float]:
    scale = rng.uniform(0.58, 0.86)
    w = size * scale * rng.uniform(0.78, 1.08)
    h = size * scale * rng.uniform(0.78, 1.12)
    cx = size * rng.uniform(0.45, 0.55)
    cy = size * rng.uniform(0.45, 0.56)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def draw_pin(layer: Image.Image, fill: tuple[int, int, int, int], rng: random.Random) -> None:
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = icon_box(rng, layer.width)
    w = x1 - x0
    h = y1 - y0
    draw.ellipse((x0 + w * 0.12, y0, x1 - w * 0.12, y0 + h * 0.66), fill=fill)
    draw.polygon([(x0 + w * 0.20, y0 + h * 0.48), (x1 - w * 0.20, y0 + h * 0.48), (x0 + w * 0.50, y1)], fill=fill)
    if rng.random() < 0.82:
        r = min(w, h) * rng.uniform(0.11, 0.18)
        cx = x0 + w * 0.50
        cy = y0 + h * rng.uniform(0.27, 0.36)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 0))


def draw_star(layer: Image.Image, fill: tuple[int, int, int, int], rng: random.Random) -> None:
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = icon_box(rng, layer.width)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    outer = min(x1 - x0, y1 - y0) * 0.48
    inner = outer * rng.uniform(0.40, 0.52)
    points = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = -math.pi / 2 + i * math.pi / 5
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    draw.polygon(points, fill=fill)


def draw_heart(layer: Image.Image, fill: tuple[int, int, int, int], rng: random.Random) -> None:
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = icon_box(rng, layer.width)
    w = x1 - x0
    h = y1 - y0
    draw.ellipse((x0 + w * 0.03, y0 + h * 0.06, x0 + w * 0.52, y0 + h * 0.55), fill=fill)
    draw.ellipse((x0 + w * 0.48, y0 + h * 0.06, x0 + w * 0.97, y0 + h * 0.55), fill=fill)
    draw.polygon([(x0 + w * 0.04, y0 + h * 0.38), (x1 - w * 0.04, y0 + h * 0.38), (x0 + w * 0.50, y1)], fill=fill)


def draw_shield(layer: Image.Image, fill: tuple[int, int, int, int], rng: random.Random) -> None:
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = icon_box(rng, layer.width)
    w = x1 - x0
    h = y1 - y0
    draw.polygon(
        [
            (x0 + w * 0.50, y0),
            (x1, y0 + h * 0.18),
            (x1 - w * 0.10, y0 + h * 0.68),
            (x0 + w * 0.50, y1),
            (x0 + w * 0.10, y0 + h * 0.68),
            (x0, y0 + h * 0.18),
        ],
        fill=fill,
    )


def draw_tag(layer: Image.Image, fill: tuple[int, int, int, int], rng: random.Random) -> None:
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = icon_box(rng, layer.width)
    w = x1 - x0
    h = y1 - y0
    draw.polygon(
        [
            (x0 + w * 0.02, y0 + h * 0.22),
            (x0 + w * 0.62, y0),
            (x1, y0 + h * 0.36),
            (x0 + w * 0.38, y1),
            (x0, y0 + h * 0.62),
        ],
        fill=fill,
    )
    if rng.random() < 0.85:
        r = min(w, h) * rng.uniform(0.06, 0.11)
        cx = x0 + w * 0.68
        cy = y0 + h * 0.24
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 0))


def draw_play(layer: Image.Image, fill: tuple[int, int, int, int], rng: random.Random) -> None:
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = icon_box(rng, layer.width)
    draw.polygon([(x0, y0), (x1, (y0 + y1) / 2), (x0, y1)], fill=fill)


def draw_bookmark(layer: Image.Image, fill: tuple[int, int, int, int], rng: random.Random) -> None:
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = icon_box(rng, layer.width)
    w = x1 - x0
    h = y1 - y0
    radius = int(min(w, h) * 0.08)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=fill)
    draw.polygon([(x0, y0 + h * 0.68), (x0 + w * 0.50, y0 + h * 0.48), (x1, y0 + h * 0.68), (x1, y1), (x0, y1)], fill=(0, 0, 0, 0))


def draw_bell(layer: Image.Image, fill: tuple[int, int, int, int], rng: random.Random) -> None:
    draw = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = icon_box(rng, layer.width)
    w = x1 - x0
    h = y1 - y0
    draw.ellipse((x0 + w * 0.36, y0, x0 + w * 0.64, y0 + h * 0.24), fill=fill)
    draw.rounded_rectangle((x0 + w * 0.16, y0 + h * 0.18, x1 - w * 0.16, y0 + h * 0.74), radius=int(w * 0.22), fill=fill)
    draw.rectangle((x0 + w * 0.06, y0 + h * 0.70, x1 - w * 0.06, y0 + h * 0.82), fill=fill)
    draw.ellipse((x0 + w * 0.39, y0 + h * 0.78, x0 + w * 0.61, y1), fill=fill)


SHAPES: list[tuple[str, ShapeDrawer]] = [
    ("pin", draw_pin),
    ("star", draw_star),
    ("heart", draw_heart),
    ("shield", draw_shield),
    ("tag", draw_tag),
    ("play", draw_play),
    ("bookmark", draw_bookmark),
    ("bell", draw_bell),
]


def smooth_background(values: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(values.astype(np.float32), (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def normalize_unit(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo, hi = np.percentile(values, [5, 96])
    if hi <= lo + 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
