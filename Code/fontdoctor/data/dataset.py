"""CFDefect-2M datasets for the three-stage FontDoctor training protocol.

Manifest format (one JSON object per line, ``*.jsonl``)::

    {
      "id": "font023_6C38_syn",        # unique sample id
      "font_id": "font023",            # font style id (font-disjoint splits)
      "char": "永",                # unicode character
      "image_path": "images/font023/6C38.png",
      "svg": "<svg ...>...</svg>",     # query glyph SVG XML (or "svg_path")
      "target_svg": "<svg ...>",       # S + A: defect-annotated target
      "has_defect": true,
      "stroke_count": 5,
      "source": "synthetic",           # or "real"
      "split": "train"
    }

Stage tasks (paper Algorithm 2):
* Stage I   (SOR): (rendered SVG code image)            -> S
* Stage II  (RVG): (raster glyph image)                 -> S
* Stage III (DET): (glyph, query SVG, +/- K references) -> skeleton(S + A)

Few-shot template-referenced vs. non-referenced universal modes are mixed
stochastically: a Bernoulli variable alpha ~ B(lambda) decides whether the
same-style reference set R_s is included (paper Sec. 4.2, Stage III).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset, Sampler

from ..configuration_fontdoctor import FontDoctorConfig
from ..constants import EOV_TOKEN, SOV_TOKEN
from .svg_renderer import render_svg_text_image
from .svg_utils import svg_to_skeleton


# ---------------------------------------------------------------------------
# Instruction templates (paper Table "Content templates", Detection Paradigms)
# ---------------------------------------------------------------------------

SYSTEM_REFERENCED = (
    "You are a helpful assistant, based on the given reference images of the "
    "font style and their corresponding SVG optical compression images, the "
    "SVG XML code is parsed from the SVG optical compression image as the "
    "style prior. At the same time, the corresponding content character "
    "images and their SVG XML code are obtained as the content branch. The "
    "model takes these two parts as joint inputs, performing contrastive "
    "learning on a stroke-by-stroke, structure-by-structure basis in a "
    "unified feature space. The differences between the two are modeled from "
    "multiple dimensions, including geometric shape, topological "
    "connectivity, contour smoothness, and detail consistency, to learn fine "
    "defects in the content glyph."
)

SYSTEM_UNIVERSAL = (
    "You are a helpful assistant, without relying on any additional style "
    "reference templates, based solely on the given image of the character "
    "to be detected and its SVG optical compression image, the corresponding "
    "SVG XML code is first parsed and recovered from the SVG optical "
    "compression image. The image and SVG XML are then jointly modeled, "
    "using the model's internally learned generic defect patterns to "
    "precisely annotate the defect locations in the SVG of the character to "
    "be detected."
)

SYSTEM_SOR = (
    "You are a helpful assistant. Read the rendered SVG XML code image and "
    "recover the exact original SVG XML text."
)

SYSTEM_RVG = (
    "You are a helpful assistant. Vectorize the given raster glyph image "
    "into its SVG XML representation with absolute coordinates."
)

USER_REFERENCED = (
    "Style reference image set: {REF_IMGS}; style reference SVG optical "
    "compression image set: {REF_SVGS}; glyph image to be detected: "
    "{QRY_IMG}; glyph SVG optical compression image to be detected: {QRY_SVG}"
)

USER_UNIVERSAL = (
    "Glyph image to be detected: {QRY_IMG}; glyph SVG optical compression "
    "image to be detected: {QRY_SVG}"
)

USER_SOR = "SVG optical compression image: {QRY_SVG}"
USER_RVG = "Glyph image: {QRY_IMG}"


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------

@dataclass
class ManifestRecord:
    id: str
    font_id: str
    char: str
    image_path: str
    svg: str
    target_svg: Optional[str]
    has_defect: bool
    stroke_count: int
    source: str
    split: str


def load_manifest(path: str) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            svg = obj.get("svg")
            if svg is None and "svg_path" in obj:
                svg = Path(obj["svg_path"]).read_text(encoding="utf-8")
            records.append(
                ManifestRecord(
                    id=obj["id"],
                    font_id=obj["font_id"],
                    char=obj.get("char", ""),
                    image_path=obj["image_path"],
                    svg=svg,
                    target_svg=obj.get("target_svg"),
                    has_defect=bool(obj.get("has_defect", False)),
                    stroke_count=int(obj.get("stroke_count", 0)),
                    source=obj.get("source", "synthetic"),
                    split=obj.get("split", "train"),
                )
            )
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CFDefect2MDataset(Dataset):
    """One dataset class serving all three training stages.

    Args:
        manifest_path : JSONL manifest of the split to use.
        image_root    : root directory of raster glyph images.
        config        : :class:`FontDoctorConfig`.
        stage         : 1 (SOR) | 2 (RVG) | 3 (DET).
    """

    def __init__(
        self,
        manifest_path: str,
        image_root: str,
        config: FontDoctorConfig,
        stage: int,
    ) -> None:
        super().__init__()
        self.records = load_manifest(manifest_path)
        self.image_root = Path(image_root)
        self.config = config
        self.stage = stage

        # same-style defect-free reference pool R_s (Stage III only)
        self._reference_pool: Dict[str, List[int]] = {}
        if stage == 3:
            for idx, rec in enumerate(self.records):
                if not rec.has_defect:
                    self._reference_pool.setdefault(rec.font_id, []).append(idx)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rec: ManifestRecord) -> Image.Image:
        return Image.open(self.image_root / rec.image_path).convert("RGB")

    def _render_svg_image(self, svg: str) -> Image.Image:
        """Phi: SVG XML text -> code image (paper Algorithm 1)."""
        return render_svg_text_image(
            svg,
            max_width=self.config.svg_render_max_width,
            font_size=self.config.svg_render_font_size,
            line_spacing=self.config.svg_render_line_spacing,
        )

    # ------------------------------------------------------------------ #
    def _build_stage1(self, rec: ManifestRecord) -> dict:
        """SOR: (Z_svg) -> S, calibrating the optical branch (paper 4.2-I)."""
        return {
            "task": "sor",
            "system": SYSTEM_SOR,
            "user": USER_SOR,
            "target": f"{SOV_TOKEN}\n{rec.svg}\n{EOV_TOKEN}",
            "query_image": None,
            "query_svg_image": self._render_svg_image(rec.svg),
            "ref_images": [],
            "ref_svg_images": [],
            "coord_targets": None,
            "stroke_count": rec.stroke_count,
        }

    def _build_stage2(self, rec: ManifestRecord) -> dict:
        """RVG: (x_v) -> S, raster-to-vector alignment (paper 4.2-II)."""
        return {
            "task": "rvg",
            "system": SYSTEM_RVG,
            "user": USER_RVG,
            "target": f"{SOV_TOKEN}\n{rec.svg}\n{EOV_TOKEN}",
            "query_image": self._load_image(rec),
            "query_svg_image": None,
            "ref_images": [],
            "ref_svg_images": [],
            "coord_targets": None,
            "stroke_count": rec.stroke_count,
        }

    def _build_stage3(self, rec: ManifestRecord) -> dict:
        """DET: defect detection with CCED skeleton target (paper 4.2-III)."""
        # Bernoulli task mixing: referenced mode iff alpha < lambda AND the
        # font style owns a non-empty reference pool.
        use_referenced = (
            random.random() < self.config.referenced_mode_prob
            and len(self._reference_pool.get(rec.font_id, [])) > 0
        )

        ref_images: List[Image.Image] = []
        ref_svg_images: List[Image.Image] = []
        if use_referenced:
            k = random.randint(1, self.config.max_reference_templates)
            pool = self._reference_pool[rec.font_id]
            ref_idx = random.sample(pool, k=min(k, len(pool)))
            for j in ref_idx:
                ref = self.records[j]
                ref_images.append(self._load_image(ref))
                ref_svg_images.append(self._render_svg_image(ref.svg))

        # S_tgt = S (+) A  ->  skeleton with <coord> placeholders
        target_svg = rec.target_svg if rec.target_svg is not None else rec.svg
        skeleton, coords = svg_to_skeleton(target_svg)
        norm = float(self.config.coord_normalize)
        coord_targets = [(x / norm, y / norm) for x, y in coords]

        return {
            "task": "det",
            "system": SYSTEM_REFERENCED if use_referenced else SYSTEM_UNIVERSAL,
            "user": USER_REFERENCED if use_referenced else USER_UNIVERSAL,
            "target": f"{SOV_TOKEN}\n{skeleton}\n{EOV_TOKEN}",
            "query_image": self._load_image(rec),
            "query_svg_image": self._render_svg_image(rec.svg),
            "ref_images": ref_images,
            "ref_svg_images": ref_svg_images,
            "coord_targets": coord_targets,
            "stroke_count": rec.stroke_count,
        }

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        if self.stage == 1:
            return self._build_stage1(rec)
        if self.stage == 2:
            return self._build_stage2(rec)
        if self.stage == 3:
            return self._build_stage3(rec)
        raise ValueError(f"unknown stage {self.stage}")


# ---------------------------------------------------------------------------
# Stroke-count curriculum (paper Algorithm 2, Stage III)
# ---------------------------------------------------------------------------

class StrokeCountCurriculumSampler(Sampler):
    """Yield sample indices sorted by ascending stroke count.

    Characters with more strokes contain longer Bézier path sequences; the
    curriculum moves from simple to complex structures (paper Sec. 4.2).
    ``chunk_shuffle`` optionally shuffles inside fixed-size windows to keep a
    minimum of stochasticity.
    """

    def __init__(self, dataset: CFDefect2MDataset, chunk_shuffle: int = 0):
        self.order = sorted(range(len(dataset)), key=lambda i: dataset.records[i].stroke_count)
        self.chunk_shuffle = chunk_shuffle

    def __iter__(self):
        order = list(self.order)
        if self.chunk_shuffle and self.chunk_shuffle > 1:
            for start in range(0, len(order), self.chunk_shuffle):
                chunk = order[start : start + self.chunk_shuffle]
                random.shuffle(chunk)
                order[start : start + self.chunk_shuffle] = chunk
        return iter(order)

    def __len__(self) -> int:
        return len(self.order)
