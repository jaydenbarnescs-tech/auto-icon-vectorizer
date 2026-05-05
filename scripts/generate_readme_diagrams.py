from __future__ import annotations

import textwrap
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto_icon_vectorizer import vectorize as adapter
from auto_icon_vectorizer import vectorize_icon_crop

EXAMPLES = ROOT / "examples"
SIZE = 128
SCALE = 4
AI_UI_SNAPSHOT = EXAMPLES / "ai-generated-ui-snapshot.png"
AI_UI_ICON_CROP = EXAMPLES / "ai-generated-ui-icon-crop.png"
AI_UI_ICON_BOX = (185, 280, 330, 425)
AI_UI_WIDE_VIEW_BOX = (80, 95, 1550, 520)
AI_UI_CARD_VIEW_BOX = (130, 210, 630, 455)


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
    ai_snapshot, ai_crop = load_ai_generated_ui_assets()
    ai_result = vectorize_icon_crop(ai_crop)
    ai_rendered_rgba = trace.render_svg_transparent(ai_result["svg"], SIZE)
    write_ai_website_workflow_diagram(ai_snapshot, ai_crop, ai_rendered_rgba, ai_result)
    write_ai_website_integration_diagram(ai_snapshot, ai_rendered_rgba)
    write_tracing_limitations_diagram(trace)


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


def make_plain_blurry_chat_crop() -> Image.Image:
    image = Image.new("RGB", (220, 160), (249, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 20, 202, 140), radius=18, fill=(255, 255, 253), outline=(224, 224, 219), width=1)
    draw.ellipse((55, 26, 151, 122), fill=(244, 235, 215))
    icon = (188, 142, 54)
    draw.ellipse((82, 50, 125, 93), outline=icon, width=5)
    draw.rectangle((93, 64, 116, 81), outline=icon, width=4)
    draw.line((98, 81, 91, 90), fill=icon, width=4)
    draw.line((82, 51, 127, 95), fill=icon, width=5)
    return ai_blur(image, blur=0.85, jpeg_quality=62)


def make_plain_low_contrast_check_crop() -> Image.Image:
    image = Image.new("RGB", (220, 160), (245, 242, 234))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((22, 19, 198, 141), radius=18, fill=(251, 249, 244), outline=(226, 222, 212), width=1)
    draw.ellipse((56, 24, 152, 120), fill=(238, 229, 208))
    icon = (190, 147, 56)
    draw.ellipse((80, 47, 128, 95), outline=icon, width=5)
    draw.line((92, 73, 103, 84, 123, 60), fill=icon, width=7, joint="curve")
    return ai_blur(image, blur=0.72, jpeg_quality=66)


def make_plain_blurry_filled_tag_crop() -> Image.Image:
    image = Image.new("RGB", (220, 160), (246, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 18, 200, 142), radius=18, fill=(255, 255, 252), outline=(224, 228, 220), width=1)
    draw.ellipse((55, 26, 151, 122), fill=(244, 235, 215))
    icon = (188, 142, 54)
    draw.polygon([(69, 78), (108, 48), (151, 58), (140, 100), (100, 130)], fill=icon)
    draw.ellipse((120, 65, 133, 78), fill=(244, 235, 215))
    draw.arc((125, 31, 174, 80), 202, 30, fill=icon, width=5)
    draw.line((162, 47, 174, 36), fill=icon, width=4)
    return ai_blur(image, blur=0.95, jpeg_quality=58)


def ai_blur(image: Image.Image, *, blur: float, jpeg_quality: int) -> Image.Image:
    import io

    small = image.resize((92, 67), Image.Resampling.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((220, 160), Image.Resampling.BICUBIC)


def load_ai_generated_ui_assets() -> tuple[Image.Image, Image.Image]:
    if not AI_UI_SNAPSHOT.exists():
        raise FileNotFoundError(
            f"{AI_UI_SNAPSHOT} is required. It is a checked-in AI-generated UI snapshot used by the README diagrams."
        )
    snapshot = Image.open(AI_UI_SNAPSHOT).convert("RGB")
    crop = snapshot.crop(AI_UI_ICON_BOX)
    crop.save(AI_UI_ICON_CROP)
    return snapshot, crop


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


def write_ai_website_workflow_diagram(snapshot: Image.Image, crop: Image.Image, icon: Image.Image, result: dict) -> None:
    width, height = 1720, 820
    image = Image.new("RGB", (width, height), "#f7f8f6")
    draw = ImageDraw.Draw(image)
    fonts = load_fonts()
    draw.text((54, 44), "Example: Recover An Icon During Website Generation", fill="#161a18", font=fonts["title"])
    draw_wrapped_text(
        draw,
        (56, 94),
        "A generated website mockup contains a style-matched but blurry raster icon. The website builder crops that icon, sends only the crop to Auto Icon Vectorizer, then inserts the returned inline SVG HTML into the final page.",
        128,
        fonts["body"],
        "#4b5350",
        27,
    )

    cards = [
        ("1. AI-generated UI", "The icon matches the mockup style, but it only exists as blurry pixels.", "mockup"),
        ("2. Detected icon crop", "A separate detection step passes one small crop to this project.", "crop"),
        ("3. Vectorizer output", "The mask is recovered and traced into transparent SVG paths.", "vector"),
        ("4. Final HTML page", "The generated site receives a scalable inline SVG instead of a raster icon.", "final"),
    ]
    x_positions = [48, 456, 864, 1272]
    for index, (title, body, kind) in enumerate(cards):
        x = x_positions[index]
        draw_card(draw, (x, 170, x + 360, 700), title, body, fonts)
        if kind == "mockup":
            panel = ai_generated_ui_panel(snapshot, (286, 190), final_icon=None, view_box=AI_UI_CARD_VIEW_BOX)
            image.paste(panel, (x + 37, 335))
        elif kind == "crop":
            draw.rounded_rectangle((x + 82, 324, x + 278, 520), radius=10, fill="#ffffff", outline="#cbd1cb", width=2)
            image.paste(crop.resize((184, 184), Image.Resampling.LANCZOS), (x + 88, 330))
            draw.text((x + 86, 555), "crop from generated UI", fill="#3d4642", font=fonts["small"])
        elif kind == "vector":
            draw.rounded_rectangle((x + 82, 322, x + 278, 518), radius=10, fill="#ffffff", outline="#cbd1cb", width=2)
            checker = checkerboard((184, 184))
            checker.alpha_composite(icon.resize((132, 132), Image.Resampling.LANCZOS), (26, 26))
            image.paste(checker.convert("RGB"), (x + 88, 328))
            draw_code_box(
                draw,
                (x + 50, 548, x + 310, 650),
                [
                    'result["svg"]',
                    'result["html"]',
                    f'{result["diagnostics"]["selectedMaskMode"]} mask',
                ],
                fonts["mono"],
            )
        else:
            panel = ai_generated_ui_panel(snapshot, (286, 190), final_icon=icon, view_box=AI_UI_CARD_VIEW_BOX)
            image.paste(panel, (x + 37, 335))
        if index < len(cards) - 1:
            draw_arrow(draw, (x + 374, 430), (x + 402, 430))

    draw.text(
        (56, 742),
        "The icon detector and page generator live outside this package. Auto Icon Vectorizer owns the cropped-icon-to-SVG/HTML step.",
        fill="#3d4642",
        font=fonts["small"],
    )
    image.save(EXAMPLES / "ai-website-icon-workflow.png")


def write_ai_website_integration_diagram(snapshot: Image.Image, icon: Image.Image) -> None:
    width, height = 1600, 760
    image = Image.new("RGB", (width, height), "#f8f7f3")
    draw = ImageDraw.Draw(image)
    fonts = load_fonts()
    draw.text((54, 44), "What The Website Builder Gets Back", fill="#161a18", font=fonts["title"])
    draw_wrapped_text(
        draw,
        (56, 94),
        "The final page can keep the generated design language while replacing a low-quality icon bitmap with transparent, color-controllable SVG HTML.",
        120,
        fonts["body"],
        "#4b5350",
        27,
    )

    draw_card(draw, (70, 165, 730, 650), "Before: Raster Icon From AI Image", "The page generator may detect layout correctly, but the icon remains a small blurred crop inside the generated screenshot.", fonts)
    before = ai_generated_ui_panel(snapshot, (520, 255), final_icon=None, view_box=AI_UI_WIDE_VIEW_BOX)
    image.paste(before, (140, 365))

    draw_card(draw, (870, 165, 1530, 650), "After: Inline SVG In The HTML", "The reconstructed page inserts a scalable SVG path where the raster icon crop was used.", fonts)
    after = ai_generated_ui_panel(snapshot, (520, 255), final_icon=icon, view_box=AI_UI_WIDE_VIEW_BOX)
    image.paste(after, (940, 365))

    draw_arrow(draw, (744, 422), (856, 422))
    draw.text((766, 456), "vectorize", fill="#66706c", font=fonts["small"])

    draw_code_box(
        draw,
        (930, 310, 1492, 370),
        [
            'result["html"] -> <span><svg>...</svg></span>',
        ],
        fonts["mono"],
    )
    image.save(EXAMPLES / "ai-website-html-integration.png")


def write_tracing_limitations_diagram(trace) -> None:
    cases = [
        ("Plain card, blurred line icon", make_plain_blurry_chat_crop()),
        ("Plain card, low-contrast stroke", make_plain_low_contrast_check_crop()),
        ("Plain card, blurred filled icon", make_plain_blurry_filled_tag_crop()),
    ]
    width, height = 1740, 1140
    image = Image.new("RGB", (width, height), "#f8f7f3")
    draw = ImageDraw.Draw(image)
    fonts = load_fonts()
    draw.text((54, 42), "Why Tracer-First Algorithms Fail On Blurry Icons", fill="#161a18", font=fonts["title"])
    draw_wrapped_text(
        draw,
        (56, 92),
        "These examples use ordinary card-style UI backgrounds. The hard part is still that the icon is a tiny blurred raster object. Tracer-first methods must guess a bitmap or color layer before tracing; this project recovers the icon mask first.",
        130,
        fonts["body"],
        "#4b5350",
        27,
    )
    columns = [
        ("Input crop", "Blurry icon on a normal UI card"),
        ("Tracer-first: binary", "Otsu threshold + Potrace, like a one-click bitmap trace"),
        ("Tracer-first: palette", "Color-layer style trace keeps halos or weak blobs"),
        ("This project", "U-Net mask recovery, cleanup, then Potrace"),
    ]
    x_positions = [70, 470, 870, 1270]
    for x, (title, subtitle) in zip(x_positions, columns):
        draw.text((x, 165), title, fill="#161a18", font=fonts["heading"])
        draw_wrapped_text(draw, (x, 198), subtitle, 31, fonts["small"], "#56605c", 24)

    row_y = [305, 585, 865]
    for (case_name, crop), y in zip(cases, row_y):
        draw.text((70, y - 38), case_name, fill="#3d4642", font=fonts["small"])
        panels = [
            crop.resize((190, 138), Image.Resampling.LANCZOS),
            trace_blurry_crop_direct(trace, crop, "threshold"),
            trace_blurry_crop_direct(trace, crop, "palette"),
            trace_blurry_crop_auto(trace, crop),
        ]
        for x, panel in zip(x_positions, panels):
            draw.rounded_rectangle((x - 10, y - 10, x + 250, y + 206), radius=12, fill="#ffffff", outline="#d5dcd5", width=2)
            if panel.mode == "RGBA":
                checker = checkerboard((220, 176))
                placed = panel.resize((150, 150), Image.Resampling.LANCZOS)
                checker.alpha_composite(placed, (35, 13))
                image.paste(checker.convert("RGB"), (x + 10, y + 16))
            else:
                framed = panel.resize((220, 176), Image.Resampling.LANCZOS)
                image.paste(framed, (x + 10, y + 16))

    draw_wrapped_text(
        draw,
        (56, 1090),
        "Manual design-tool tracing can still be useful for one-off cleanup. The automation problem is that a website generator cannot pause for a human to choose the best threshold, delete card halos, reconnect blurry strokes, and re-export every icon.",
        150,
        fonts["small"],
        "#3d4642",
        24,
    )
    image.save(EXAMPLES / "tracing-alone-failure-modes.png")


def ai_generated_ui_panel(
    snapshot: Image.Image,
    size: tuple[int, int],
    *,
    final_icon: Image.Image | None,
    view_box: tuple[int, int, int, int],
) -> Image.Image:
    source = snapshot.crop(view_box).convert("RGB")
    panel = source.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    draw = ImageDraw.Draw(panel)
    mapped = map_box(AI_UI_ICON_BOX, view_box, size)
    if final_icon is None:
        draw.rounded_rectangle(mapped, radius=6, outline=(41, 49, 44, 220), width=3)
        label_w = min(140, size[0] - mapped[0] - 8)
        label_box = (mapped[0], max(4, mapped[1] - 26), mapped[0] + label_w, max(28, mapped[1] - 4))
        draw.rounded_rectangle(label_box, radius=5, fill=(24, 30, 27, 215))
        draw.text((label_box[0] + 7, label_box[1] + 3), "detected crop", fill=(235, 238, 232, 255), font=load_fonts()["small"])
    else:
        clear_icon_region(panel, mapped)
        paste_vector_icon(panel, final_icon, mapped)
        draw.rounded_rectangle(mapped, radius=6, outline=(207, 166, 72, 235), width=3)
        label_w = min(142, size[0] - mapped[0] - 8)
        label_box = (mapped[0], max(4, mapped[1] - 26), mapped[0] + label_w, max(28, mapped[1] - 4))
        draw.rounded_rectangle(label_box, radius=5, fill=(207, 166, 72, 230))
        draw.text((label_box[0] + 7, label_box[1] + 3), "inline SVG", fill=(31, 33, 28, 255), font=load_fonts()["small"])
    return panel.convert("RGB")


def map_box(box: tuple[int, int, int, int], view_box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    vx0, vy0, vx1, vy1 = view_box
    sx = size[0] / max(1, vx1 - vx0)
    sy = size[1] / max(1, vy1 - vy0)
    return (
        round((x0 - vx0) * sx),
        round((y0 - vy0) * sy),
        round((x1 - vx0) * sx),
        round((y1 - vy0) * sy),
    )


def clear_icon_region(panel: Image.Image, box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(panel)
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    radius = max(18, min(x1 - x0, y1 - y0) // 2 - 5)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(245, 237, 220, 255),
        outline=(231, 217, 188, 255),
        width=1,
    )


def paste_vector_icon(panel: Image.Image, icon: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    side = max(18, min(x1 - x0, y1 - y0) - 28)
    placed = icon.resize((side, side), Image.Resampling.LANCZOS)
    px = (x0 + x1 - side) // 2
    py = (y0 + y1 - side) // 2
    panel.alpha_composite(placed, (px, py))


def trace_blurry_crop_auto(trace, crop: Image.Image) -> Image.Image:
    try:
        result = vectorize_icon_crop(crop, icon_color="#b8892f")
        return trace.render_svg_transparent(result["svg"], SIZE)
    except Exception:
        return failure_panel_rgba("trace failed")


def trace_blurry_crop_direct(trace, crop: Image.Image, mode: str) -> Image.Image:
    try:
        source = trace.canonical_image(crop.convert("RGB"), SIZE)
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if mode == "threshold":
            bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
            dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
            mask = bright if border_touch_ratio(bright) <= border_touch_ratio(dark) else dark
            mask = cv2.medianBlur(mask.astype(np.uint8) * 255, 3) > 0
        elif mode == "edges":
            edges = cv2.Canny(gray, 35, 100)
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.dilate(edges, kernel, iterations=1) > 0
        elif mode == "palette":
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            border = np.zeros(gray.shape, dtype=bool)
            border[:8, :] = True
            border[-8:, :] = True
            border[:, :8] = True
            border[:, -8:] = True
            bg = np.median(lab[border], axis=0)
            dist = np.linalg.norm(lab - bg[None, None, :], axis=2)
            # Approximate a palette/color-layer vectorizer: anything far enough
            # from the card background becomes traceable foreground. This often
            # keeps normal UI icon containers and blur halos, not only the icon.
            mask = dist > max(8.0, float(np.percentile(dist, 76)))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        else:
            raise ValueError(mode)
        mask = trace.preprocess_mask_for_potrace(mask.astype(np.uint8))
        if int(mask.sum()) <= 0:
            return failure_panel_rgba("empty mask")
        options = trace.potrace_options(float(trace.estimate_mask_thickness(mask)), "default")
        svg = trace.normalize_svg(
            trace.trace_with_potrace(trace.mask_to_trace_bitmap(mask), "#111827", options),
            "#111827",
            SIZE,
        )
        return trace.render_svg_transparent(svg, SIZE)
    except Exception:
        return failure_panel_rgba("trace failed")


def border_touch_ratio(mask: np.ndarray) -> float:
    border = np.zeros(mask.shape, dtype=bool)
    border[:6, :] = True
    border[-6:, :] = True
    border[:, :6] = True
    border[:, -6:] = True
    return float((mask & border).sum()) / max(1.0, float(mask.sum()))


def failure_panel_rgba(text: str) -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    font = load_fonts()["small"]
    draw.text((18, 54), text, fill=(116, 41, 41, 255), font=font)
    return image


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


def checkerboard(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    cell = 14
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2 == 0:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(227, 231, 226, 255))
    return image


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
