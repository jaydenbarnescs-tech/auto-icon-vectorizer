from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Union

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIZE = 128
SCALE = 4
RUNTIME_DIR = ROOT / "auto_icon_vectorizer" / "runtime"


Color = Union[tuple[int, int, int], int]
DrawFn = Callable[[ImageDraw.ImageDraw, Color, random.Random], None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a public synthetic stroke-training corpus for the gated U-Net training script."
    )
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--out-dir", type=Path, default=RUNTIME_DIR / "truth-stress-eval")
    args = parser.parse_args()

    reports = generate_reports(args.count, args.seed, args.out_dir)
    payload = {
        "description": "Synthetic public corpus for retraining the stroke gated U-Net branch on single-color icon foreground masks.",
        "foregroundContract": "one cropped icon, one dominant foreground color, one binary alpha truth mask",
        "count": len(reports),
        "reports": reports,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "latest-run.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(reports), "manifest": str((args.out_dir / "latest-run.json").resolve())}, indent=2))


def generate_reports(count: int, seed: int, out_dir: Path) -> list[dict[str, str]]:
    images_dir = out_dir / "images"
    truth_dir = out_dir / "truth"
    images_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, str]] = []
    icons: list[tuple[str, DrawFn]] = [
        ("check-circle", draw_check_circle),
        ("chat-slash", draw_chat_slash),
        ("cube", draw_cube),
        ("cart", draw_cart),
        ("list", draw_list),
        ("storefront", draw_storefront),
        ("bell", draw_bell),
        ("search", draw_search),
    ]
    backgrounds = ["dark", "light", "stripes", "dots", "gradient", "noisy"]

    for index in range(count):
        rng = random.Random(seed + index * 7919)
        icon_name, draw_icon = icons[index % len(icons)]
        bg_mode = backgrounds[(index // len(icons)) % len(backgrounds)]
        source, truth = build_case(draw_icon, bg_mode, rng)
        split = "s02" if index % 5 == 0 else "s01"
        ident = f"{split}-{index:03d}"
        source_path = images_dir / f"{ident}-{icon_name}.png"
        truth_path = truth_dir / f"{ident}-{icon_name}.png"
        source.save(source_path)
        truth.save(truth_path)
        reports.append(
            {
                "id": ident,
                "icon": icon_name,
                "backgroundMode": bg_mode,
                "sourceCrop": str(source_path.resolve()),
                "truthIcon": str(truth_path.resolve()),
            }
        )
    return reports


def build_case(draw_icon: DrawFn, bg_mode: str, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    size = SIZE * SCALE
    source = make_background(size, bg_mode, rng)
    truth = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    source_draw = ImageDraw.Draw(source)
    truth_draw = ImageDraw.Draw(truth)
    color = rng.choice(
        [
            (210, 190, 146),
            (225, 218, 188),
            (70, 130, 92),
            (207, 87, 55),
            (91, 154, 180),
            (184, 141, 64),
        ]
    )
    draw_icon(source_draw, color, rng)
    draw_icon(truth_draw, (255, 255, 255), rng)

    blur = rng.choice([0.0, 0.0, 0.4, 0.7])
    if blur:
        source = source.filter(ImageFilter.GaussianBlur(blur))
    source = source.resize((SIZE, SIZE), Image.Resampling.LANCZOS).convert("RGB")
    truth = truth.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    alpha = truth.getchannel("A").point(lambda value: 255 if value > 20 else 0)
    truth = Image.merge("RGBA", [Image.new("L", (SIZE, SIZE), 255)] * 3 + [alpha])
    return source, truth


def make_background(size: int, mode: str, rng: random.Random) -> Image.Image:
    if mode == "light":
        base = rng.choice([(238, 234, 220), (229, 236, 232), (236, 231, 240)])
    else:
        base = rng.choice([(18, 21, 18), (27, 31, 29), (35, 34, 28), (44, 39, 54)])
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)
    if mode == "stripes":
        for x in range(-size, size * 2, rng.randint(34, 54)):
            draw.line([(x, 0), (x + size, size)], fill=shift(base, rng, 18), width=rng.randint(10, 22))
    elif mode == "dots":
        for _ in range(80):
            x = rng.randrange(size)
            y = rng.randrange(size)
            r = rng.randint(3, 13)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=shift(base, rng, 30))
    elif mode == "gradient":
        for y in range(size):
            t = y / max(1, size - 1)
            c = tuple(int(base[channel] * (1 - t) + shift(base, rng, 28)[channel] * t) for channel in range(3))
            draw.line([(0, y), (size, y)], fill=c)
    elif mode == "noisy":
        for _ in range(120):
            x = rng.randrange(size)
            y = rng.randrange(size)
            r = rng.randint(2, 10)
            draw.rectangle([x - r, y - r, x + r, y + r], fill=shift(base, rng, 35))
    return image


def shift(color: tuple[int, int, int], rng: random.Random, amount: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, channel + rng.randint(-amount, amount))) for channel in color)


def p(value: float) -> int:
    return round(value * SCALE)


def width(rng: random.Random, base: int = 5) -> int:
    return max(3, (base + rng.choice([-1, 0, 1, 2])) * SCALE)


def draw_check_circle(draw: ImageDraw.ImageDraw, color: Color, rng: random.Random) -> None:
    draw.ellipse([p(22), p(18), p(106), p(102)], outline=color, width=width(rng))
    draw.line([(p(43), p(60)), (p(57), p(73)), (p(85), p(42))], fill=color, width=width(rng, 7), joint="curve")


def draw_chat_slash(draw: ImageDraw.ImageDraw, color: Color, rng: random.Random) -> None:
    draw.ellipse([p(15), p(15), p(113), p(113)], outline=color, width=width(rng))
    draw.rectangle([p(39), p(43), p(89), p(75)], outline=color, width=width(rng))
    draw.line([(p(47), p(75)), (p(37), p(88))], fill=color, width=width(rng))
    draw.line([(p(30), p(29)), (p(99), p(99))], fill=color, width=width(rng, 6))


def draw_cube(draw: ImageDraw.ImageDraw, color: Color, rng: random.Random) -> None:
    pts = [(p(31), p(38)), (p(64), p(20)), (p(98), p(38)), (p(98), p(82)), (p(64), p(105)), (p(30), p(82))]
    draw.line(pts + [pts[0]], fill=color, width=width(rng), joint="curve")
    draw.line([(p(31), p(38)), (p(64), p(57)), (p(98), p(38))], fill=color, width=width(rng))
    draw.line([(p(64), p(57)), (p(64), p(105))], fill=color, width=width(rng))
    draw.line([(p(47), p(29)), (p(81), p(48))], fill=color, width=width(rng))


def draw_cart(draw: ImageDraw.ImageDraw, color: Color, rng: random.Random) -> None:
    draw.line([(p(24), p(36)), (p(37), p(36)), (p(47), p(76)), (p(91), p(76)), (p(101), p(46)), (p(43), p(46))], fill=color, width=width(rng), joint="curve")
    draw.ellipse([p(48), p(86), p(61), p(99)], outline=color, width=width(rng))
    draw.ellipse([p(82), p(86), p(95), p(99)], outline=color, width=width(rng))


def draw_list(draw: ImageDraw.ImageDraw, color: Color, rng: random.Random) -> None:
    for y in [38, 61, 84]:
        draw.ellipse([p(29), p(y - 4), p(37), p(y + 4)], fill=color)
        draw.line([(p(49), p(y)), (p(96), p(y))], fill=color, width=width(rng))


def draw_storefront(draw: ImageDraw.ImageDraw, color: Color, rng: random.Random) -> None:
    draw.rectangle([p(30), p(55), p(98), p(101)], outline=color, width=width(rng))
    draw.line([(p(25), p(55)), (p(36), p(30)), (p(92), p(30)), (p(103), p(55))], fill=color, width=width(rng), joint="curve")
    for x in [42, 56, 70, 84]:
        draw.line([(p(x), p(31)), (p(x - 5), p(55))], fill=color, width=width(rng, 4))


def draw_bell(draw: ImageDraw.ImageDraw, color: Color, rng: random.Random) -> None:
    draw.arc([p(37), p(32), p(91), p(91)], 190, 350, fill=color, width=width(rng))
    draw.line([(p(39), p(64)), (p(31), p(87)), (p(98), p(87)), (p(89), p(64))], fill=color, width=width(rng), joint="curve")
    draw.arc([p(55), p(84), p(73), p(105)], 10, 170, fill=color, width=width(rng))


def draw_search(draw: ImageDraw.ImageDraw, color: Color, rng: random.Random) -> None:
    draw.ellipse([p(32), p(27), p(78), p(73)], outline=color, width=width(rng, 6))
    draw.line([(p(70), p(68)), (p(99), p(97))], fill=color, width=width(rng, 7), joint="curve")


if __name__ == "__main__":
    main()
