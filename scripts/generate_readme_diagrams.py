from __future__ import annotations

import textwrap
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_icon_vectorizer import vectorize_icon_crop

EXAMPLES = ROOT / "examples"
SIZE = 128
SCALE = 4


def main() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    crop = make_demo_crop()
    prefix = EXAMPLES / "diagram-icon"
    result = vectorize_icon_crop(crop, output_prefix=prefix, write_html_artifact=True)
    source = Image.open(str(prefix) + "-source.png").convert("RGB")
    mask = Image.open(str(prefix) + "-mask.png").convert("RGB")
    rendered = Image.open(str(prefix) + "-rendered.png").convert("RGB")
    write_pipeline_diagram(source, mask, rendered, result)
    write_output_contract_diagram(result)


def make_demo_crop() -> Image.Image:
    image = Image.new("RGB", (220, 160), (29, 31, 26))
    draw = ImageDraw.Draw(image)
    for x in range(-80, 300, 26):
        color = (45, 49, 42) if (x // 26) % 2 == 0 else (35, 38, 33)
        draw.line([(x, 0), (x + 160, 160)], fill=color, width=9)
    color = (211, 192, 150)
    draw.ellipse([54, 24, 166, 136], outline=color, width=7)
    draw.rectangle([88, 66, 139, 100], outline=color, width=7)
    draw.line([(96, 100), (83, 116)], fill=color, width=7)
    draw.line([(69, 46), (155, 126)], fill=color, width=8)
    return image


def write_pipeline_diagram(source: Image.Image, mask: Image.Image, rendered: Image.Image, result: dict) -> None:
    width, height = 1680, 760
    image = Image.new("RGB", (width, height), "#f7f8f6")
    draw = ImageDraw.Draw(image)
    fonts = load_fonts()
    draw.text((54, 44), "From Icon Crop To Inline SVG HTML", fill="#161a18", font=fonts["title"])
    draw_wrapped_text(
        draw,
        (56, 92),
        "The model accepts an arbitrary-size crop, normalizes it internally, recovers the foreground mask, traces it, and returns SVG plus an HTML wrapper.",
        126,
        fonts["body"],
        "#4b5350",
        27,
    )

    cards = [
        ("1. Cropped input", "Any crop size. The icon should already be centered or tightly cropped.", source),
        ("2. Foreground mask", "The learned mask removes noisy background pixels before tracing.", mask),
        ("3. SVG path", "Potrace converts the cleaned binary mask into scalable Bezier paths.", rendered),
        ("4. HTML wrapper", "The same SVG is returned inside a span with CSS and metadata hooks.", None),
    ]
    x_positions = [58, 458, 858, 1258]
    for index, (title, body, panel) in enumerate(cards):
        x = x_positions[index]
        draw_card(draw, (x, 154, x + 350, 650), title, body, fonts)
        if panel is not None:
            framed = panel.resize((210, 210), Image.Resampling.NEAREST)
            image.paste(framed, (x + 70, 330))
        else:
            draw_code_box(
                draw,
                (x + 34, 318, x + 316, 560),
                [
                    '<span class="vector-icon"',
                    '  data-vectorizer="auto-...">',
                    '  <svg viewBox="0 0 128 128"',
                    '       aria-hidden="true">',
                    '    <path d="M..." />',
                    "  </svg>",
                    "</span>",
                ],
                fonts["mono"],
            )
        if index < len(cards) - 1:
            draw_arrow(draw, (x + 365, 402), (x + 395, 402))

    draw.text((56, 690), f"Default renderer: {result['diagnostics']['pipelineRenderer']}", fill="#3d4642", font=fonts["small"])
    draw.text((56, 718), "CLI prints SVG by default. Python returns both result['svg'] and result['html'].", fill="#3d4642", font=fonts["small"])
    image.save(EXAMPLES / "pipeline-diagram.png")


def write_output_contract_diagram(result: dict) -> None:
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), "#f8f7f3")
    draw = ImageDraw.Draw(image)
    fonts = load_fonts()
    draw.text((54, 44), "Why SVG And HTML Are Both Returned", fill="#161a18", font=fonts["title"])
    draw_wrapped_text(
        draw,
        (56, 92),
        "SVG is the portable graphic. HTML is the ready-to-insert package for web pages, with styling, metadata, and accessibility hooks around that same SVG.",
        112,
        fonts["body"],
        "#4b5350",
        27,
    )

    draw_card(draw, (68, 160, 702, 650), "Raw SVG", "Best when saving an asset file, sending SVG to another tool, or embedding it manually.", fonts)
    draw_code_box(
        draw,
        (108, 330, 662, 550),
        [
            '<svg width="128" height="128"',
            '     viewBox="0 0 128 128">',
            '  <path fill="#d3c096"',
            '        d="M..." />',
            "</svg>",
        ],
        fonts["mono"],
    )
    draw.text((108, 586), "CLI default stdout: result['svg']", fill="#2f3a36", font=fonts["body_bold"])

    draw_card(draw, (798, 160, 1432, 650), "Inline SVG HTML", "Best when code needs one DOM-ready string with CSS class, source id, renderer metadata, title, or aria label.", fonts)
    draw_code_box(
        draw,
        (838, 330, 1392, 580),
        [
            '<span class="vector-icon"',
            '      data-source-id="settings"',
            '      data-vectorizer="auto-...">',
            '  <svg role="img" aria-label="Settings">',
            '    <path fill="#d3c096" d="M..." />',
            "  </svg>",
            "</span>",
        ],
        fonts["mono"],
    )
    draw.text((838, 616), "Use --stdout html or result['html']", fill="#2f3a36", font=fonts["body_bold"])

    draw_arrow(draw, (710, 402), (790, 402))
    draw.text((678, 430), "same traced SVG", fill="#66706c", font=fonts["small"])
    image.save(EXAMPLES / "output-contract.png")


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, fonts: dict) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#d9ded9", width=2)
    draw.text((x0 + 28, y0 + 28), title, fill="#161a18", font=fonts["heading"])
    y = y0 + 74
    for line in wrap(body, 34):
        draw.text((x0 + 28, y), line, fill="#56605c", font=fonts["small"])
        y += 25


def draw_code_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], font: ImageFont.ImageFont) -> None:
    draw.rounded_rectangle(box, radius=8, fill="#181c1a", outline="#313936", width=2)
    x0, y0, _x1, _y1 = box
    y = y0 + 24
    for line in lines:
        draw.text((x0 + 22, y), line, fill="#dce7df", font=font)
        y += 28


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill="#52605a", width=4)
    ex, ey = end
    draw.polygon([(ex, ey), (ex - 12, ey - 8), (ex - 12, ey + 8)], fill="#52605a")


def wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    font: ImageFont.ImageFont,
    fill: str,
    line_height: int,
) -> None:
    x, y = xy
    for line in wrap(text, width):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def load_fonts() -> dict:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_path = next((path for path in candidates if Path(path).exists()), None)
    if font_path is None:
        default = ImageFont.load_default()
        return {name: default for name in ["title", "heading", "body", "body_bold", "small", "mono"]}
    return {
        "title": ImageFont.truetype(font_path, 36),
        "heading": ImageFont.truetype(font_path, 25),
        "body": ImageFont.truetype(font_path, 21),
        "body_bold": ImageFont.truetype(font_path, 22),
        "small": ImageFont.truetype(font_path, 18),
        "mono": ImageFont.truetype(font_path, 18),
    }


if __name__ == "__main__":
    main()
