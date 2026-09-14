"""Rendering utilities of FontDoctor.

* :func:`render_svg_text_image` -- the dynamic rendering function Phi of
  paper Algorithm 1 ("SVG Optical Compression"): an SVG XML string is drawn
  as a monospace code-editor view on a white canvas of width <= 512 px, the
  right-side whitespace of short lines is cropped, and the canvas height
  grows linearly with the number of wrapped text lines.
* :func:`rasterize_svg` -- rasterize an SVG document into a grayscale glyph
  image (used for the MSE / IoU evaluation and dataset construction).
"""

import os
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Algorithm 1: SVG XML text -> code image
# ---------------------------------------------------------------------------

_DEFAULT_MONO_CANDIDATES = [
    "DejaVuSansMono.ttf",
    "consola.ttf",
    "cour.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/cour.ttf",
]


def load_monospace_font(font_size: int = 14, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """Load a monospace font F_mono for the code-editor rendering."""
    if font_path is not None:
        return ImageFont.truetype(font_path, font_size)
    for candidate in _DEFAULT_MONO_CANDIDATES:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, font_size)
            except (OSError, ValueError):
                continue
    return ImageFont.load_default()


def _wrap_line(line: str, draw: ImageDraw.ImageDraw, font, max_width: int) -> List[str]:
    """Greedy character-level wrapping so no rendered line exceeds max_width."""
    if draw.textlength(line, font=font) <= max_width:
        return [line]
    wrapped, buf = [], ""
    for ch in line:
        if draw.textlength(buf + ch, font=font) > max_width and buf:
            wrapped.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        wrapped.append(buf)
    return wrapped


def render_svg_text_image(
    svg_text: str,
    max_width: int = 512,
    font_size: int = 14,
    line_spacing: int = 4,
    font_path: Optional[str] = None,
    padding: int = 4,
) -> Image.Image:
    """Rendering function Phi of paper Algorithm 1.

    Args:
        svg_text    : SVG XML string S.
        max_width   : W_max = 512 px canvas-width cap.
        font_size   : monospace font size.
        line_spacing: extra pixels between lines.
        font_path   : optional explicit monospace font file.
        padding     : canvas padding in pixels.

    Returns:
        An ``(H, W)`` RGB image; W = min(text width, max_width), H grows
        linearly with the number of (wrapped) text lines.
    """
    font = load_monospace_font(font_size, font_path)

    # measure with a scratch canvas
    scratch = Image.new("RGB", (8, 8), (255, 255, 255))
    draw = ImageDraw.Draw(scratch)

    usable_width = max_width - 2 * padding
    lines: List[str] = []
    for raw_line in svg_text.splitlines() or [svg_text]:
        lines.extend(_wrap_line(raw_line, draw, font, usable_width))

    ascent, descent = font.getmetrics()
    line_height = ascent + descent + line_spacing

    # W <- w_text (cropped) if the text is narrower than W_max, else W_max
    text_width = max((draw.textlength(line, font=font) for line in lines), default=0)
    width = min(int(text_width) + 2 * padding, max_width)
    height = line_height * len(lines) + 2 * padding

    # white canvas of all ones, then rasterize the text (Algorithm 1)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += line_height
    return image


# ---------------------------------------------------------------------------
# SVG document -> raster glyph image
# ---------------------------------------------------------------------------

def rasterize_svg(
    svg_text: str,
    output_size: Tuple[int, int] = (512, 512),
    background: int = 255,
) -> Image.Image:
    """Rasterize an SVG document to a grayscale image.

    Prefers ``cairosvg`` (accurate); falls back to a minimal PIL polygon
    renderer based on sampled path polylines when cairosvg is unavailable.
    """
    try:
        import cairosvg
        import io

        png_bytes = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=output_size[0],
            output_height=output_size[1],
            background_color="white",
        )
        return Image.open(io.BytesIO(png_bytes)).convert("L")
    except ImportError:
        pass

    # Fallback: polyline sampling + filled polygons (approximate).
    from .svg_utils import sample_path_points, split_glyph_and_defect_paths

    image = Image.new("L", output_size, background)
    draw = ImageDraw.Draw(image)
    glyph_ds, defect_ds = split_glyph_and_defect_paths(svg_text)
    for d in glyph_ds:
        pts = sample_path_points(d)
        if len(pts) >= 3:
            draw.polygon(pts, fill=0)
    for d in defect_ds:
        pts = sample_path_points(d)
        if len(pts) >= 3:
            draw.polygon(pts, fill=128)
    return image


def rasterize_defect_map(
    svg_text: str, output_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """Rasterize *only* the defect-annotation paths as a binary map.

    Used by the raster-domain evaluation (MSE between the predicted defect
    map I_pred and the ground-truth defect map I_target).
    """
    from .svg_utils import sample_path_points, split_glyph_and_defect_paths

    image = Image.new("L", output_size, 0)
    draw = ImageDraw.Draw(image)
    _, defect_ds = split_glyph_and_defect_paths(svg_text)
    for d in defect_ds:
        pts = sample_path_points(d)
        if len(pts) >= 3:
            draw.polygon(pts, fill=255, outline=255)
        elif len(pts) >= 2:
            draw.line(pts, fill=255, width=3)
        elif len(pts) == 1:
            draw.point(pts[0], fill=255)
    return image
