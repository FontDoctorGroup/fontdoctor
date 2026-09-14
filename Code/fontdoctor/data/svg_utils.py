"""SVG utilities for FontDoctor.

Covers the vector-side representation of the paper:

* seven drawing commands (M / L / H / V / Q / C / Z) with absolute
  coordinates (paper Appendix "Data Processing and Curation");
* integer coordinate quantization (round every float parameter to an int,
  reducing tokenizer length, paper Sec. 3.1);
* defect annotation groups ``<g class="defect"><path .../></g>`` whose fill
  is red (paper Appendix "Defect Annotation");
* conversion between a full SVG target string S_tgt and its CCED skeleton
  S_bar in which every 2-D coordinate pair is replaced by ``<coord>``
  (paper Sec. 3.6), plus the inverse coordinate fill-back used at inference;
* polyline sampling of Bézier paths for the vector-domain metrics
  (DTW / LDTW / Chamfer distance).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..constants import COORD_TOKEN, COMMAND_ARITY, DEFECT_FILL_RGB, DEFECT_GROUP_CLASS

# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"([MLHVCQZmlhvcqz])|(-?\d+(?:\.\d+)?)")


@dataclass
class PathSegment:
    """One absolute drawing command with its 2-D coordinate pairs."""

    command: str
    points: List[Tuple[float, float]] = field(default_factory=list)


def parse_path_d(d: str) -> List[PathSegment]:
    """Parse an SVG path ``d`` string into absolute segments.

    Relative commands are converted to absolute ones and implicit command
    repetition (e.g. ``M 0 0 1 1``) is expanded.
    """
    tokens: List = []
    for m in _TOKEN_RE.finditer(d):
        cmd, num = m.group(1), m.group(2)
        tokens.append(cmd if cmd is not None else float(num))

    segments: List[PathSegment] = []
    i = 0
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    active_cmd: Optional[str] = None

    def _read_point(idx: int) -> Tuple[Tuple[float, float], int]:
        return (float(tokens[idx]), float(tokens[idx + 1])), idx + 2

    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            active_cmd = tok
            i += 1
            if active_cmd in ("Z", "z"):
                segments.append(PathSegment("Z", []))
                cur = start
                active_cmd = None
            continue

        assert active_cmd is not None, f"path data starts with a number: {d[:40]}..."
        cmd = active_cmd
        relative = cmd.islower()
        cmd = cmd.upper()
        arity = COMMAND_ARITY[cmd]

        # implicit repetition: the first moveto pair is M, the rest become L
        if cmd == "M" and segments and segments[-1].command == "M":
            cmd = "L"
            arity = 1

        pts: List[Tuple[float, float]] = []
        for _ in range(arity):
            if cmd == "H":
                x = float(tokens[i]); i += 1
                pt = (x + cur[0] if relative else x, cur[1])
            elif cmd == "V":
                y = float(tokens[i]); i += 1
                pt = (cur[0], y + cur[1] if relative else y)
            else:
                pt, i = _read_point(i)
                if relative:
                    pt = (pt[0] + cur[0], pt[1] + cur[1])
            pts.append(pt)

        segments.append(PathSegment(cmd, pts))
        cur = pts[-1]
        if cmd == "M":
            start = cur

    return segments


def segments_to_path_d(segments: Sequence[PathSegment], quantize: bool = True) -> str:
    """Serialize absolute segments back to a path ``d`` string.

    H/V are normalized to L (FontDoctor keeps 5 absolute command types in
    practice); every parameter is rounded to an integer when ``quantize``.
    """

    def _fmt(v: float) -> str:
        if quantize:
            return str(int(round(v)))
        return f"{v:g}"

    parts: List[str] = []
    for seg in segments:
        cmd = "L" if seg.command in ("H", "V") else seg.command
        parts.append(cmd)
        if cmd == "Z":
            continue
        arity = COMMAND_ARITY.get(seg.command, len(seg.points))
        pts = seg.points[: arity if seg.command not in ("H", "V") else 1]
        for x, y in pts:
            parts.append(f"{_fmt(x)} {_fmt(y)}")
    return " ".join(parts)


def quantize_path_d(d: str) -> str:
    """Round every path parameter to an integer (paper Sec. 3.1)."""
    return segments_to_path_d(parse_path_d(d), quantize=True)


# ---------------------------------------------------------------------------
# Glyph SVG documents
# ---------------------------------------------------------------------------

SVG_HEADER = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
    'viewBox="0 0 {w} {h}">'
)


def build_glyph_svg(
    path_ds: Sequence[str],
    defect_path_ds: Optional[Sequence[str]] = None,
    width: int = 512,
    height: int = 512,
    quantize: bool = True,
) -> str:
    """Assemble the canonical FontDoctor glyph SVG document.

    Normal glyph contours are plain ``<path>`` elements; defective regions are
    appended as ``<g class="defect">`` groups with red-filled paths, following
    the expert annotation format (paper Fig. 3 & Appendix "Defect Annotation").
    """
    if quantize:
        path_ds = [quantize_path_d(d) for d in path_ds]
        if defect_path_ds:
            defect_path_ds = [quantize_path_d(d) for d in defect_path_ds]

    parts = [SVG_HEADER.format(w=width, h=height), "<g>"]
    for d in path_ds:
        parts.append(f'<path d="{d}"></path>')
    parts.append("</g>")
    for idx, d in enumerate(defect_path_ds or []):
        r, g_, b = DEFECT_FILL_RGB
        parts.append(f'<g class="{DEFECT_GROUP_CLASS}" id="{idx:02d}">')
        parts.append(f'<path d="{d}" fill="rgb({r},{g_},{b})"></path>')
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


_PATH_D_RE = re.compile(r'<path[^>]*\bd="([^"]*)"[^>]*>')
_DEFECT_GROUP_RE = re.compile(
    rf'<g\s+class="{DEFECT_GROUP_CLASS}"[^>]*>(.*?)</g>', re.DOTALL
)


def split_glyph_and_defect_paths(svg: str) -> Tuple[List[str], List[str]]:
    """Split an SVG document into (glyph paths, defect-annotation paths)."""
    defect_ds: List[str] = []
    for group in _DEFECT_GROUP_RE.findall(svg):
        defect_ds.extend(_PATH_D_RE.findall(group))
    glyph_svg = _DEFECT_GROUP_RE.sub("", svg)
    glyph_ds = _PATH_D_RE.findall(glyph_svg)
    return glyph_ds, defect_ds


def has_defect(svg: str) -> bool:
    return f'class="{DEFECT_GROUP_CLASS}"' in svg


# ---------------------------------------------------------------------------
# CCED skeleton conversion (paper Sec. 3.6)
# ---------------------------------------------------------------------------

def svg_to_skeleton(svg: str) -> Tuple[str, List[Tuple[float, float]]]:
    """Replace every 2-D coordinate pair in path data by ``<coord>``.

    Returns the structural skeleton S_bar (SVG tags, path commands and
    ``<coord>`` placeholders) and the list of ground-truth coordinates
    (normalized to [0, 1] by the caller when building targets).
    """

    coords: List[Tuple[float, float]] = []

    def _replace_d(match: re.Match) -> str:
        prefix, d = match.group(1), match.group(2)
        segments = parse_path_d(d)
        parts: List[str] = []
        for seg in segments:
            parts.append(seg.command)
            if seg.command == "Z":
                continue
            arity = COMMAND_ARITY.get(seg.command, len(seg.points))
            for pt in seg.points[:arity]:
                coords.append((float(pt[0]), float(pt[1])))
                parts.append(COORD_TOKEN)
        return prefix + " ".join(parts) + '"'

    skeleton = re.sub(r'(\bd=")([^"]*)"', _replace_d, svg)
    return skeleton, coords


def skeleton_to_svg(
    skeleton: str, coords: Sequence[Tuple[float, float]], quantize: bool = True
) -> str:
    """Fill predicted coordinates back into a skeleton (inference-time)."""
    coord_iter = iter(coords)

    def _fmt(v: float) -> str:
        return str(int(round(v))) if quantize else f"{v:g}"

    def _fill(match: re.Match) -> str:
        prefix, d = match.group(1), match.group(2)
        out_tokens: List[str] = []
        for tok in d.split(" "):
            if tok == COORD_TOKEN:
                try:
                    x, y = next(coord_iter)
                except StopIteration as exc:
                    raise ValueError("not enough coordinates to fill the skeleton") from exc
                out_tokens.append(f"{_fmt(x)} {_fmt(y)}")
            else:
                out_tokens.append(tok)
        return prefix + " ".join(out_tokens) + '"'

    return re.sub(r'(\bd=")([^"]*)"', _fill, skeleton)


# ---------------------------------------------------------------------------
# Bézier polyline sampling (for DTW / LDTW / Chamfer metrics)
# ---------------------------------------------------------------------------

def _sample_quadratic(p0, p1, p2, n: int) -> List[Tuple[float, float]]:
    return [
        (
            (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1],
        )
        for t in (i / (n - 1) for i in range(n))
    ]


def _sample_cubic(p0, p1, p2, p3, n: int) -> List[Tuple[float, float]]:
    pts = []
    for i in range(n):
        t = i / (n - 1)
        mt = 1 - t
        pts.append(
            (
                mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0],
                mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1],
            )
        )
    return pts


def sample_path_points(d: str, samples_per_segment: int = 16) -> List[Tuple[float, float]]:
    """Sample a polyline approximation of a path's contour points."""
    segments = parse_path_d(d)
    points: List[Tuple[float, float]] = []
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    for seg in segments:
        if seg.command == "M":
            cur = seg.points[0]
            start = cur
            points.append(cur)
        elif seg.command in ("L", "H", "V"):
            for pt in seg.points:
                for i in range(1, samples_per_segment):
                    t = i / samples_per_segment
                    points.append((cur[0] + t * (pt[0] - cur[0]), cur[1] + t * (pt[1] - cur[1])))
                points.append(pt)
                cur = pt
        elif seg.command == "Q":
            p1, p2 = seg.points
            pts = _sample_quadratic(cur, p1, p2, samples_per_segment)
            points.extend(pts[1:])
            cur = p2
        elif seg.command == "C":
            p1, p2, p3 = seg.points
            pts = _sample_cubic(cur, p1, p2, p3, samples_per_segment)
            points.extend(pts[1:])
            cur = p3
        elif seg.command == "Z":
            if cur != start:
                for i in range(1, samples_per_segment):
                    t = i / samples_per_segment
                    points.append((cur[0] + t * (start[0] - cur[0]), cur[1] + t * (start[1] - cur[1])))
                points.append(start)
            cur = start
    return points


def sample_svg_defect_points(svg: str, samples_per_segment: int = 16) -> List[Tuple[float, float]]:
    """Sample contour points of *all* defect-annotation paths of a document."""
    _, defect_ds = split_glyph_and_defect_paths(svg)
    points: List[Tuple[float, float]] = []
    for d in defect_ds:
        points.extend(sample_path_points(d, samples_per_segment))
    return points
