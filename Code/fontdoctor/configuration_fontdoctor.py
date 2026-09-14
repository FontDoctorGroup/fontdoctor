"""FontDoctor configuration.

Values follow the paper:
* Character vision encoder  : Qwen2.5-VL-style ViT (hidden 1280, 32 layers,
  16 heads, FFN 3456, patch 14, window 112, full attention {7, 15, 23, 31}),
  native input resolution, 2 x 2 spatial merge, out dim 2560.
* OCR optical encoder       : DeepEncoder of DeepSeek-OCR (SAM ViT-B + 16x
  conv compressor + CLIP-L), frozen, dynamic-resolution tiling.
* Language model            : Qwen3-4B (hidden 2560, 36 layers, 32 heads,
  8 KV heads, head dim 128, FFN 9728, vocab 151936, context 40960,
  rope theta 1e6) -- identical to the official Qwen/Qwen3-4B config.
* DeepStack injection       : features from 3 ViT layers injected into the
  first 3 decoder blocks (paper Sec. 3.5; ablation: L_inj = 3 is optimal).
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CharViTConfig:
    """Configuration of the native-resolution character vision encoder.

    Mirrors ``Qwen2_5_VLVisionConfig`` of the official Qwen2.5-VL
    (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) with the merger
    output re-targeted to the Qwen3-4B hidden size (2560).
    """

    hidden_size: int = 1280
    depth: int = 32
    num_heads: int = 16
    intermediate_size: int = 3456
    hidden_act: str = "silu"
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    window_size: int = 112
    fullatt_block_indexes: Tuple[int, ...] = (7, 15, 23, 31)
    out_hidden_size: int = 2560  # language hidden dim D (Qwen3-4B)
    # ViT layers whose features are projected and injected into the decoder
    # (DeepStack).  Default: three global-attention layers.
    deepstack_layer_indexes: Tuple[int, ...] = (7, 15, 23)
    rope_theta: float = 10000.0


@dataclass
class OCREncoderConfig:
    """Configuration of the frozen SVG optical-compression branch."""

    image_size: int = 1024           # global view resolution  (Base mode)
    patch_image_size: int = 640      # local crop resolution   (Gundam mode)
    min_crops: int = 2
    max_crops: int = 9
    fused_dim: int = 2048            # 1024 (CLIP) + 1024 (SAM) after concat
    n_embed: int = 2560              # == language hidden dim D
    freeze: bool = True              # E_ocr is frozen in ALL stages (paper)


@dataclass
class LanguageConfig:
    """Official Qwen/Qwen3-4B backbone configuration."""

    model_name_or_path: str = "Qwen/Qwen3-4B"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6


@dataclass
class FontDoctorConfig:
    """Top-level FontDoctor configuration."""

    char_vit: CharViTConfig = field(default_factory=CharViTConfig)
    ocr: OCREncoderConfig = field(default_factory=OCREncoderConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)

    # Decoder blocks receiving residual visual injection (l1, l2, l3).
    deepstack_decoder_layers: Tuple[int, ...] = (0, 1, 2)

    # CCED: L = L_tok + lambda_coord * L_coord
    lambda_coord: float = 1.0
    coord_hidden_ratio: int = 1        # hidden = ratio * D inside coord MLP
    coord_normalize: float = 512.0     # coords are regressed in [0, 1] / 512

    # Few-shot template-referenced detection.
    max_reference_templates: int = 6   # K (saturates at K = 6, paper Appendix)
    referenced_mode_prob: float = 0.5  # lambda of the Bernoulli task mixing

    # Rendering of the SVG XML code image (Algorithm 1).
    svg_render_max_width: int = 512
    svg_render_font_size: int = 14
    svg_render_line_spacing: int = 4
