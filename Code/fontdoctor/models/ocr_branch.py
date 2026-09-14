"""SVG Optical-Compression vision branch (E_ocr + g_ocr) of FontDoctor.

The branch is built around the *frozen* DeepEncoder of the official
DeepSeek-OCR (https://github.com/deepseek-ai/DeepSeek-OCR,
vendored in ``third_party/deepencoder.py``):

    SAM ViT-B  ->  16x convolutional compressor  ->  CLIP-L
    local crops (n x 640 x 640) + one global view (1024 x 1024)

The dynamic-resolution tiling (``find_closest_aspect_ratio`` /
``dynamic_preprocess``) and the feature fusion flow
(concat CLIP tokens with SAM features, append an ``image_newline`` token per
feature row, and terminate with a ``view_seperator``) are adapted from the
official ``modeling_deepseekocr.py``.

Differences w.r.t. the official model (paper Sec. 3.4):
* the DeepSeek text decoder is discarded -- only the OCR visual tower is kept;
* the fused 2048-d features F_svg are projected into the Qwen3 hidden space
  by a lightweight two-layer MLP g_ocr (paper Eq. 7), which is the *only*
  trainable part of this branch.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageOps

from ..configuration_fontdoctor import OCREncoderConfig
from ..data.image_transforms import SimpleImageTransform
from .third_party.deepencoder import build_clip_l, build_sam_vit_b


# ---------------------------------------------------------------------------
# Dynamic-resolution preprocessing (adapted from official modeling_deepseekocr)
# ---------------------------------------------------------------------------
def find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios, width: int, height: int, image_size: int
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image, min_num: int = 2, max_num: int = 9, image_size: int = 640
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """InternVL-style tiling used by DeepSeek-OCR (Gundam mode).

    Returns a list of ``n`` local crops and the crop grid ``(w_num, h_num)``.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    return processed_images, target_aspect_ratio


class BasicImageTransform(SimpleImageTransform):
    """Official normalization of DeepSeek-OCR (mean = std = 0.5)."""

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Optical Compression Encoder
# ---------------------------------------------------------------------------
class OpticalCompressionEncoder(nn.Module):
    """Frozen DeepEncoder + trainable OCR projector g_ocr.

    forward(patches, global_view, crop_shape) -> Z_svg of shape
    ``(K_svg, n_embed)`` where ``K_svg`` adapts to the rendered code-image
    resolution and the number of dynamic crops (paper Eq. 6-7).
    """

    def __init__(self, config: OCREncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Frozen OCR visual tower (official DeepEncoder).
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        # g_ocr: lightweight two-layer MLP projector into the LM hidden space.
        self.projector = nn.Sequential(
            nn.Linear(config.fused_dim, config.n_embed),
            nn.GELU(),
            nn.Linear(config.n_embed, config.n_embed),
        )

        embed_std = 1.0 / torch.sqrt(torch.tensor(config.n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(config.n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(config.n_embed) * embed_std)

        self.image_transform = BasicImageTransform()

        if config.freeze:
            for p in self.sam_model.parameters():
                p.requires_grad_(False)
            for p in self.vision_model.parameters():
                p.requires_grad_(False)
            self.sam_model.eval()
            self.vision_model.eval()

    # -- frozen DeepEncoder inference -------------------------------------
    def _encode_tiles(self, images: torch.Tensor) -> torch.Tensor:
        """SAM -> 16x compressor -> CLIP, then 2048-d feature fusion."""
        with torch.no_grad():
            sam_feats = self.sam_model(images)                      # (B, 1024, h, w)
            clip_feats = self.vision_model(images, sam_feats)       # (B, 1 + h*w, 1024)
            fused = torch.cat(
                (clip_feats[:, 1:], sam_feats.flatten(2).permute(0, 2, 1)), dim=-1
            )                                                       # (B, h*w, 2048)
        return fused

    def encode(
        self,
        local_crops: Optional[torch.Tensor],
        global_view: torch.Tensor,
        crop_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Encode one rendered SVG code image into OCR visual tokens Z_svg.

        Args:
            local_crops : ``(n, 3, 640, 640)`` dynamic crops, or ``None`` when
                the image degrades to the Base mode (both sides < 640).
            global_view : ``(1, 3, 1024, 1024)`` padded global view.
            crop_shape  : ``(width_crop_num, height_crop_num)`` of the tiles.

        Returns:
            ``(K_svg, n_embed)`` tensor on the module device.
        """
        dtype = self.projector[0].weight.dtype
        device = self.projector[0].weight.device

        pieces: List[torch.Tensor] = []
        if local_crops is not None and crop_shape is not None and local_crops.shape[0] > 0:
            local = self._encode_tiles(local_crops.to(device=device, dtype=dtype))
            local = self.projector(local)                           # (n, h2*w2, D)
            n, hw2, ndim = local.shape
            h2 = w2 = int(hw2**0.5)
            width_crop_num, height_crop_num = crop_shape
            local = (
                local.view(height_crop_num, width_crop_num, h2, w2, ndim)
                .permute(0, 2, 1, 3, 4)
                .reshape(height_crop_num * h2, width_crop_num * w2, ndim)
            )
            local = torch.cat(
                [local, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, ndim)],
                dim=1,
            )
            pieces.append(local.view(-1, ndim))

        global_feats = self._encode_tiles(global_view.to(device=device, dtype=dtype))
        global_feats = self.projector(global_feats)                 # (1, h*w, D)
        _, hw, ndim = global_feats.shape
        h = w = int(hw**0.5)
        global_feats = global_feats.view(h, w, ndim)
        global_feats = torch.cat(
            [global_feats, self.image_newline[None, None, :].expand(h, 1, ndim)], dim=1
        )
        pieces.append(global_feats.view(-1, ndim))

        pieces.append(self.view_seperator[None, :])
        return torch.cat(pieces, dim=0)

    # -- image preparation -------------------------------------------------
    def prepare_image(self, image: Image.Image) -> dict:
        """PIL image -> tensors consumed by :meth:`encode`.

        Mirrors the official DeepSeek-OCR preprocessing: when both sides are
        smaller than the local-crop size, no tiling is applied (Base mode);
        otherwise 2-9 crops plus one global 1024 x 1024 view (Gundam mode).
        """
        cfg = self.config
        image = image.convert("RGB")
        w, h = image.size

        if w <= cfg.patch_image_size and h <= cfg.patch_image_size:
            # Base mode: single (padded) global view.
            global_view = self._to_global_view(image)
            return {"local_crops": None, "global_view": global_view, "crop_shape": None}

        crops, ratio = dynamic_preprocess(
            image, min_num=cfg.min_crops, max_num=cfg.max_crops, image_size=cfg.patch_image_size
        )
        local_crops = torch.stack([self.image_transform(c) for c in crops])
        global_view = self._to_global_view(image)
        return {
            "local_crops": local_crops,
            "global_view": global_view,
            "crop_shape": (int(ratio[0]), int(ratio[1])),
        }

    def _to_global_view(self, image: Image.Image) -> torch.Tensor:
        """Aspect-preserving resize + white padding to 1024 x 1024 (official)."""
        size = self.config.image_size
        padded = ImageOps.pad(image, (size, size), color=(255, 255, 255))
        return self.image_transform(padded).unsqueeze(0)

    def forward(self, prepared: dict) -> torch.Tensor:
        return self.encode(
            prepared["local_crops"], prepared["global_view"], prepared["crop_shape"]
        )

    # -- deterministic token counting --------------------------------------
    def count_tokens(self, prepared: dict) -> int:
        """Number of OCR visual tokens K_svg for a prepared image.

        The count is resolution-deterministic (no forward pass needed), which
        lets the data collator pre-expand ``<svg_img>`` placeholder tokens:
            local  : n * s^2 + ratio_h * s   (s  = crop side tokens)
            global : g^2 + g                 (g  = global side tokens)
            plus one view-separator token.
        """
        s = self.config.patch_image_size // 16 // 4   # SAM patch16 + 2 stride-2 convs
        g = self.config.image_size // 16 // 4
        total = g * g + g + 1
        if prepared.get("local_crops") is not None and prepared.get("crop_shape") is not None:
            ratio_w, ratio_h = prepared["crop_shape"]
            n = ratio_w * ratio_h
            total += n * s * s + ratio_h * s
        return total
