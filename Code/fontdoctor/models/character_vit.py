"""Native-resolution Character Vision Encoder (E_v) of FontDoctor.

This module is a faithful image-only adaptation of the official Qwen2.5-VL
vision transformer
  * HF transformers: transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
    (classes ``Qwen2_5_VisionTransformerPretrainedModel``,
    ``Qwen2_5_VLVisionBlock``, ``Qwen2_5_VLVisionSdpaAttention``,
    ``Qwen2_5_VLPatchMerger`` ...),
  * upstream repo: https://github.com/QwenLM/Qwen2.5-VL

Paper-specific extensions
-------------------------
1. Temporal handling is removed (glyph images are still images, so
   ``temporal_patch_size`` is fixed to 1).
2. ``deepstack_layer_indexes`` exposes intermediate ViT features
   F^{(s_k)} which are projected by per-level 2-layer MLP mergers
   g^{(k)} and later injected into shallow LLM decoder blocks
   (paper Sec. 3.5, "DeepStack-style multi-layer visual injection").
3. The final merger maps 4 x 1280 -> 2560 (Qwen3-4B hidden size), see
   paper Table "Network configuration of the Character Vision Encoder".
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_fontdoctor import CharViTConfig


# ---------------------------------------------------------------------------
# Basic building blocks (identical to the official Qwen2.5-VL implementation)
# ---------------------------------------------------------------------------
class Qwen2RMSNorm(nn.Module):
    """Qwen2RMSNorm is equivalent to T5LayerNorm (official Qwen2.5-VL code)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VisionMLP(nn.Module):
    """SwiGLU MLP of the Qwen2.5-VL vision tower (with bias)."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class VisionPatchEmbed(nn.Module):
    """2-D variant of the official ``Qwen2_5_VisionPatchEmbed`` (t = 1)."""

    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1280):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (num_patches_total, C, P, P) -> (num_patches_total, D)."""
        target_dtype = self.proj.weight.dtype
        return self.proj(patches.to(dtype=target_dtype)).flatten(1)


class VisionRotaryEmbedding(nn.Module):
    """Official ``Qwen2_5_VisionRotaryEmbedding``."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Official ``apply_rotary_pos_emb_vision`` of Qwen2.5-VL."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class VisionSdpaAttention(nn.Module):
    """Official ``Qwen2_5_VLVisionSdpaAttention`` (var-length via cu_seqlens)."""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class VisionBlock(nn.Module):
    """Official ``Qwen2_5_VLVisionBlock`` (RMSNorm + attention + SwiGLU MLP)."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.attn = VisionSdpaAttention(hidden_size, num_heads=num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size, bias=True)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchMerger(nn.Module):
    """Official ``Qwen2_5_VLPatchMerger``: RMSNorm + 2-layer MLP.

    Implements Eq. (4) of the paper: each 2 x 2 feature neighborhood is
    spatially concatenated and projected by a two-layer MLP, reducing the
    visual sequence length from m*n to m*n/4.
    """

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


# ---------------------------------------------------------------------------
# Character Vision Encoder
# ---------------------------------------------------------------------------
class CharacterVisionEncoder(nn.Module):
    """FontDoctor character vision encoder E_v.

    Args:
        config: :class:`CharViTConfig`.

    Inputs:
        patches  : ``(total_patches, C, P, P)`` float tensor.  Glyph images of
                   a batch are flattened into one long patch sequence, exactly
                   like the official Qwen2.5-VL batched-image handling.
        grid_thw : ``(num_images, 3)`` long tensor with (t=1, grid_h, grid_w).

    Outputs: dict with
        ``last``       : merged visual tokens  ``(total_merged, out_dim)``
        ``deepstack``  : list of per-level merged tokens for the layers in
                         ``config.deepstack_layer_indexes``.
    """

    def __init__(self, config: CharViTConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = list(config.fullatt_block_indexes)
        self.deepstack_layer_indexes = list(config.deepstack_layer_indexes)
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = VisionPatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, theta=config.rope_theta)

        self.blocks = nn.ModuleList(
            [
                VisionBlock(config.hidden_size, config.intermediate_size, config.num_heads)
                for _ in range(config.depth)
            ]
        )
        self.merger = PatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        # Per-level DeepStack mergers g^{(k)} (paper Eq. 8, two-layer MLPs).
        self.deepstack_mergers = nn.ModuleList(
            [
                PatchMerger(
                    dim=config.out_hidden_size,
                    context_dim=config.hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                )
                for _ in self.deepstack_layer_indexes
            ]
        )
        self.gradient_checkpointing = False

    # -- positional helpers (verbatim logic from the official implementation)
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h, device=grid_thw.device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w, device=grid_thw.device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = int(grid_thw[:, 1:].max())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def get_window_index(self, grid_thw: torch.Tensor):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w, device=grid_thw.device).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += grid_t * llm_grid_h * llm_grid_w
        return torch.cat(window_index, dim=0), cu_window_seqlens

    def forward(self, patches: torch.Tensor, grid_thw: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden_states = self.patch_embed(patches)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Features of DeepStack layers are collected in *windowed* order and
        # un-permuted after the loop (same reverse_indices as the main path).
        deepstack_feats: List[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing(blk, hidden_states, cu_seqlens_now, position_embeddings)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_layer_indexes:
                deepstack_feats.append(hidden_states)

        reverse_indices = torch.argsort(window_index)
        merged = self.merger(hidden_states)[reverse_indices, :]
        deepstack_merged = [
            merger(feat)[reverse_indices, :] for merger, feat in zip(self.deepstack_mergers, deepstack_feats)
        ]
        return {"last": merged, "deepstack": deepstack_merged}

    def _gradient_checkpointing(self, blk, hidden_states, cu_seqlens, position_embeddings):
        return torch.utils.checkpoint.checkpoint(
            blk.__call__, hidden_states, cu_seqlens, position_embeddings, use_reentrant=False
        )


# ---------------------------------------------------------------------------
# Image preprocessing helpers (native resolution, Qwen2.5-VL patch ordering)
# ---------------------------------------------------------------------------
def resize_to_multiple_of_patch(
    height: int, width: int, patch_size: int = 14, merge_size: int = 2
) -> Tuple[int, int]:
    """Native-resolution sizing: round H/W to multiples of patch*merge (28).

    The aspect ratio is preserved by rounding to the *nearest* multiple, as in
    the official Qwen2.5-VL ``smart_resize``.
    """
    factor = patch_size * merge_size

    def _round(x: int) -> int:
        return max(factor, int(round(x / factor)) * factor)

    return _round(height), _round(width)


def image_to_patches(
    image_tensor: torch.Tensor, patch_size: int = 14, merge_size: int = 2
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Convert a (C, H, W) image into merge-ordered patches.

    The patch order follows the official Qwen2.5-VL processor: 2 x 2 spatial
    neighborhoods are laid out adjacently so that every 4 consecutive tokens
    form one PatchMerger unit (paper Eq. 4).

    Returns:
        patches : (grid_h*grid_w, C, P, P)
        grid    : (1, grid_h, grid_w)
    """
    c, h, w = image_tensor.shape
    assert h % patch_size == 0 and w % patch_size == 0, "H/W must be multiples of patch_size"
    grid_h, grid_w = h // patch_size, w // patch_size
    assert grid_h % merge_size == 0 and grid_w % merge_size == 0, "grid must be even (merge_size=2)"

    # (C, gh, P, gw, P) -> patches with 2x2 neighborhoods adjacent:
    # official ordering: reshape(grid_h/2, 2, grid_w/2, 2) then permute.
    x = image_tensor.reshape(c, grid_h // merge_size, merge_size, patch_size,
                             grid_w // merge_size, merge_size, patch_size)
    x = x.permute(1, 4, 2, 5, 0, 3, 6)  # (gh/2, gw/2, 2, 2, C, P, P)
    x = x.reshape(grid_h * grid_w, c, patch_size, patch_size)
    return x, (1, grid_h, grid_w)
