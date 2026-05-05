from __future__ import annotations

import textwrap
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_icon_vectorizer import vectorize as adapter
from auto_icon_vectorizer import vectorize_icon_crop

EXAMPLES = ROOT / "examples"
SIZE = 128
SCALE = 4


def main() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    crop = make_demo_crop()
    crop.save(EXAMPLES / "sample-ai-icon-crop.png")
    prefix = EXAMPLES / "diagram-icon"
    result = vectorize_icon_crop(crop, output_prefix=prefix, write_html_artifact=True)
    source = Image.open(str(prefix) + "-source.png").convert("RGB")
    mask = Image.open(str(prefix) + "-mask.png").convert("RGB")
    rendered = Image.open(str(prefix) + "-rendered.png").convert("RGB")
    write_pipeline_diagram(source, mask, rendered, result)
    write_output_contract_diagram(result)
    trace = adapter._load_runtime()
    rendered_rgba = trace.render_svg_transparent(result["svg"], SIZE)
    write_transparent_background_diagram(rendered_rgba)


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


def write_transparent_background_diagram(icon: Image.Image) -> None:
    width, height = 1500, 700
    image = Image.new("RGB", (width, height), "#f8f8f4")
    draw = ImageDraw.Draw(image)
    fonts = load_fonts()
    draw.text((54, 44), "Transparent SVG Background", fill="#161a18", font=fonts["title"])
    draw_wrapped_text(
        draw,
        (56, 92),
        "The generated SVG contains only foreground paths. There is no background rectangle, so the same icon can be placed on light, dark, patterned, or gradient surfaces.",
        112,
        fonts["body"],
        "#4b5350",
        27,
    )

    panels = [
        ("Light UI", light_background),
        ("Dark UI", dark_background),
        ("Patterned", patterned_background),
        ("Gradient", gradient_background),
    ]
    x_positions = [70, 425, 780, 1135]
    for title, background_fn in panels:
        x = x_positions.pop(0)
        draw.rounded_rectangle((x, 170, x + 290, 588), radius=12, fill="#ffffff", outline="#d9ded9", width=2)
        draw.text((x + 28, 202), title, fill="#161a18", font=fonts["heading"])
        bg = background_fn((210, 210))
        composed = bg.convert("RGBA")
        placed = icon.resize((150, 150), Image.Resampling.LANCZOS)
        composed.alpha_composite(placed, ((210 - 150) // 2, (210 - 150) // 2))
        image.paste(composed.convert("RGB"), (x + 40, 292))

    draw.text(
        (56, 636),
        "Note: the background is transparent, but readability still depends on the icon color having enough contrast with the page background.",
        fill="#3d4642",
        font=fonts["small"],
    )
    image.save(EXAMPLES / "transparent-backgrounds.png")


def light_background(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, "#f5f2ea")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, size[0] - 18, size[1] - 18), radius=18, fill="#ffffff", outline="#ddd8ca", width=2)
    return image


def dark_background(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, "#111412")
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], 18):
        draw.line([(0, y), (size[0], y + 28)], fill="#202720", width=7)
    return image


def patterned_background(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, "#2d3140")
    draw = ImageDraw.Draw(image)
    for x in range(-size[0], size[0] * 2, 28):
        draw.line([(x, 0), (x + size[0], size[1])], fill="#3c4256", width=10)
    for x in range(18, size[0], 44):
        for y in range(22, size[1], 44):
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#546078")
    return image


def gradient_background(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(size[1]):
        t = y / max(1, size[1] - 1)
        r = round(28 * (1 - t) + 82 * t)
        g = round(87 * (1 - t) + 44 * t)
        b = round(101 * (1 - t) + 96 * t)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    return image


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
