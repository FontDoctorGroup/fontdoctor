"""Shared helpers for the FontDoctor entry-point scripts."""

import os
import sys

# allow running scripts directly from the repo root: python scripts/xxx.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fontdoctor.configuration_fontdoctor import FontDoctorConfig
from fontdoctor.constants import EXTRA_SPECIAL_TOKENS
from fontdoctor.models.modeling_fontdoctor import FontDoctorModel


def build_tokenizer(name_or_path: str):
    """Official Qwen3 tokenizer + FontDoctor special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": EXTRA_SPECIAL_TOKENS}
    )
    return tokenizer


def build_model(
    config: FontDoctorConfig,
    tokenizer,
    char_vit_path: str = None,
    deepseek_ocr_path: str = None,
    checkpoint: str = None,
    torch_dtype=None,
):
    """Construct FontDoctor on official backbones (or load a trained ckpt)."""
    import torch

    dtype = torch_dtype or torch.bfloat16
    model = FontDoctorModel.from_pretrained_backbones(
        config,
        char_vit_path=char_vit_path,
        deepseek_ocr_path=deepseek_ocr_path,
        torch_dtype=dtype,
    )
    # extend token embeddings for the added special tokens
    model.language_model.resize_token_embeddings(len(tokenizer))

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[common] loaded {checkpoint}: {len(missing)} missing / {len(unexpected)} unexpected keys")
    return model
