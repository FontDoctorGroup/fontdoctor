"""Batch collation: text assembly, placeholder expansion, image flattening.

The collator converts dataset samples (see :mod:`fontdoctor.data.dataset`)
into the exact tensors consumed by :class:`FontDoctorModel`:

* every raster glyph image is converted to merge-ordered ViT patches and its
  ``<glyph_img>`` placeholder is repeated ``(grid_h/2)*(grid_w/2)`` times;
* every rendered SVG code image is tiled for the OCR branch and its
  ``<svg_img>`` placeholder is repeated ``K_svg`` times (deterministic, see
  :meth:`OpticalCompressionEncoder.count_tokens`);
* the chat text follows the official Qwen3 ``<|im_start|>`` format;
* ``labels`` mask the prompt part with -100 (next-token supervision only on
  the assistant target, teacher forcing);
* ``coord_targets`` collects the normalized 2-D coordinates of all
  ``<coord>`` placeholders (CCED, paper Eq. 13).
"""

from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from ..configuration_fontdoctor import FontDoctorConfig
from ..constants import GLYPH_IMG_TOKEN, SVG_IMG_TOKEN
from ..models.character_vit import image_to_patches, resize_to_multiple_of_patch
from .image_transforms import SimpleImageTransform

# Official Qwen2.5-VL image normalization (OpenAI CLIP statistics).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_chat_prompt(
    system_text: str, user_text: str, target_text: Optional[str] = None
) -> str:
    """Qwen3 chat format (official ``<|im_start|>`` template)."""
    text = (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if target_text is not None:
        text += f"{target_text}<|im_end|>"
    return text


class FontDoctorCollator:
    """Collate FontDoctor samples for training / evaluation."""

    def __init__(self, config: FontDoctorConfig, tokenizer, ocr_encoder=None):
        self.config = config
        self.tokenizer = tokenizer
        self.ocr_encoder = ocr_encoder  # used for prepare_image + count_tokens
        self.image_transform = SimpleImageTransform(mean=CLIP_MEAN, std=CLIP_STD)
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # image handling
    # ------------------------------------------------------------------ #
    def _process_glyph_image(self, image: Image.Image):
        """Native-resolution glyph image -> merge-ordered patches + grid."""
        w, h = image.size
        new_h, new_w = resize_to_multiple_of_patch(
            h, w, self.config.char_vit.patch_size, self.config.char_vit.spatial_merge_size
        )
        if (new_h, new_w) != (h, w):
            image = image.resize((new_w, new_h), Image.BICUBIC)
        tensor = self.image_transform(image)
        patches, grid = image_to_patches(
            tensor, self.config.char_vit.patch_size, self.config.char_vit.spatial_merge_size
        )
        num_merged_tokens = (grid[1] // 2) * (grid[2] // 2)
        return patches, grid, num_merged_tokens

    def _process_svg_image(self, image: Image.Image):
        """Rendered SVG code image -> OCR tiling + deterministic K_svg."""
        if self.ocr_encoder is None:
            raise RuntimeError("collator needs `ocr_encoder` to prepare SVG images")
        prepared = self.ocr_encoder.prepare_image(image)
        return prepared, self.ocr_encoder.count_tokens(prepared)

    # ------------------------------------------------------------------ #
    # text handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _repeat(token: str, n: int) -> str:
        return " ".join([token] * n) if n > 0 else ""

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------ #
    def __call__(self, samples: Sequence[dict]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        glyph_patches_all: List[torch.Tensor] = []
        glyph_grids: List[torch.Tensor] = []
        svg_prepared_all: List[dict] = []
        coord_targets_all: List[torch.Tensor] = []

        for sample in samples:
            # ---- visual placeholders, in prompt order --------------------
            glyph_token_chunks: Dict[str, str] = {}

            ref_img_strs, ref_svg_strs = [], []
            for ref_img in sample["ref_images"]:
                patches, grid, n_tok = self._process_glyph_image(ref_img)
                glyph_patches_all.append(patches)
                glyph_grids.append(torch.tensor(grid))
                ref_img_strs.append(self._repeat(GLYPH_IMG_TOKEN, n_tok))
            for ref_svg_img in sample["ref_svg_images"]:
                prepared, k_tok = self._process_svg_image(ref_svg_img)
                svg_prepared_all.append(prepared)
                ref_svg_strs.append(self._repeat(SVG_IMG_TOKEN, k_tok))

            qry_img_str = ""
            if sample["query_image"] is not None:
                patches, grid, n_tok = self._process_glyph_image(sample["query_image"])
                glyph_patches_all.append(patches)
                glyph_grids.append(torch.tensor(grid))
                qry_img_str = self._repeat(GLYPH_IMG_TOKEN, n_tok)

            qry_svg_str = ""
            if sample["query_svg_image"] is not None:
                prepared, k_tok = self._process_svg_image(sample["query_svg_image"])
                svg_prepared_all.append(prepared)
                qry_svg_str = self._repeat(SVG_IMG_TOKEN, k_tok)

            # ---- text -----------------------------------------------------
            user_text = sample["user"].format(
                REF_IMGS=", ".join(ref_img_strs),
                REF_SVGS=", ".join(ref_svg_strs),
                QRY_IMG=qry_img_str,
                QRY_SVG=qry_svg_str,
            )
            prompt_text = build_chat_prompt(sample["system"], user_text)

            # tokenize prompt / target separately to guarantee prefix
            # consistency of the label mask (BPE does not merge across the
            # <|im_start|>assistant\n boundary).
            prompt_ids = self._encode(prompt_text)
            if sample["target"] is None:
                # generation mode: no assistant target is appended
                full_ids = prompt_ids
                labels = [-100] * len(prompt_ids)
            else:
                target_ids = self._encode(f"{sample['target']}<|im_end|>")
                full_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids

            input_ids_list.append(full_ids)
            labels_list.append(labels)

            if sample.get("coord_targets"):
                coord_targets_all.append(torch.tensor(sample["coord_targets"], dtype=torch.float32))

        # ---- padding ------------------------------------------------------
        max_len = max(len(ids) for ids in input_ids_list)
        bsz = len(input_ids_list)
        input_ids = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, (ids, lbs) in enumerate(zip(input_ids_list, labels_list)):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            labels[i, : len(lbs)] = torch.tensor(lbs)
            attention_mask[i, : len(ids)] = 1

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "glyph_patches": torch.cat(glyph_patches_all, dim=0) if glyph_patches_all else None,
            "glyph_grid_thw": torch.stack(glyph_grids) if glyph_grids else None,
            "svg_prepared": svg_prepared_all if svg_prepared_all else None,
            "coord_targets": torch.cat(coord_targets_all, dim=0) if coord_targets_all else None,
        }
        return batch
