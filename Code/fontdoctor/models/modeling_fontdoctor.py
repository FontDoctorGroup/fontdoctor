"""FontDoctor model: dual vision encoders + Qwen3 decoder + CCED.

Architecture (paper Fig. "Overview of FontDoctor")
--------------------------------------------------
* E_v    : native-resolution character vision encoder
           (:class:`CharacterVisionEncoder`, Qwen2.5-VL-style ViT).
* E_ocr  : frozen OCR optical-compression encoder
           (:class:`OpticalCompressionEncoder`, official DeepSeek-OCR
           DeepEncoder) + trainable projector g_ocr.
* M_theta: official Qwen3-4B causal LM (transformers ``Qwen3ForCausalLM``),
           receiving
             - glyph visual tokens  (from E_v, 2x2 merged),
             - SVG optical tokens   (Z_svg, from E_ocr + g_ocr),
             - text tokens,
           with DeepStack-style residual injection of multi-level ViT
           features into the first decoder blocks (paper Eq. 8).
* g_phi  : continuous coordinate regression head of the CCED
           (paper Eq. 12); trained with L = L_tok + lambda * L_coord.

Placeholder convention
----------------------
The collator emits *pre-expanded* sequences: every raster glyph occupies
``(grid_h/2) * (grid_w/2)`` consecutive ``<glyph_img>`` tokens and every
rendered SVG code image occupies ``K_svg`` consecutive ``<svg_img>`` tokens
(see :meth:`OpticalCompressionEncoder.count_tokens`).  The model replaces
those positions with the corresponding visual embeddings, exactly like the
official DeepSeek-OCR / Qwen-VL ``masked_scatter`` flow.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..configuration_fontdoctor import FontDoctorConfig
from ..constants import COORD_TOKEN, GLYPH_IMG_TOKEN, SVG_IMG_TOKEN
from .character_vit import CharacterVisionEncoder
from .ocr_branch import OpticalCompressionEncoder
from .projectors import CoordinateRegressionHead


class FontDoctorModel(nn.Module):
    def __init__(self, config: FontDoctorConfig, language_model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        self.char_vit = CharacterVisionEncoder(config.char_vit)
        self.ocr_encoder = OpticalCompressionEncoder(config.ocr)

        # Official Qwen3-4B decoder (HuggingFace transformers).
        self.language_model = language_model if language_model is not None else self._build_language_model()
        hidden = self.config.language.hidden_size

        self.coord_head = CoordinateRegressionHead(hidden, mlp_ratio=config.coord_hidden_ratio)

        # DeepStack bookkeeping: {decoder_layer_idx: (positions, features)}
        self._pending_injections: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._injection_hooks = []
        self._register_deepstack_hooks()

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _build_language_model(self) -> nn.Module:
        """Instantiate the Qwen3-4B architecture (random init).

        Use :meth:`from_pretrained_backbones` to load the official weights.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(self.config.language.model_name_or_path)
        return AutoModelForCausalLM.from_config(cfg)

    @classmethod
    def from_pretrained_backbones(
        cls,
        config: FontDoctorConfig,
        char_vit_path: Optional[str] = None,
        deepseek_ocr_path: Optional[str] = None,
        torch_dtype=torch.bfloat16,
    ) -> "FontDoctorModel":
        """Build FontDoctor on top of the official foundation-model weights.

        * Language decoder : ``Qwen/Qwen3-4B`` (official HF checkpoint).
        * Character ViT    : initialized from the official Qwen2.5-VL vision
          tower (``Qwen/Qwen2.5-VL-3B-Instruct``); the merger output layer is
          re-initialized because its out-dim changes 2048 -> 2560.
        * OCR encoder      : official DeepSeek-OCR DeepEncoder weights
          (``deepseek-ai/DeepSeek-OCR``); kept frozen.
        """
        language_model = Qwen3ForCausalLM.from_pretrained(
            config.language.model_name_or_path, torch_dtype=torch_dtype
        )
        model = cls(config, language_model=language_model)

        if char_vit_path is not None:
            model.load_char_vit_weights(char_vit_path)
        if deepseek_ocr_path is not None:
            model.load_deepencoder_weights(deepseek_ocr_path)
        return model

    # ------------------------------------------------------------------ #
    # weight loading
    # ------------------------------------------------------------------ #
    def load_char_vit_weights(self, source: str) -> List[str]:
        """Load the official Qwen2.5-VL vision tower into E_v.

        ``source`` is a HF repo id or a local directory of a Qwen2.5-VL
        checkpoint.  Keys are mapped ``visual.<name> -> <name>``; tensors with
        mismatched shapes (the re-targeted merger output layer) are skipped
        and left randomly initialized.
        """
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
        import os

        path = source if os.path.isdir(source) else snapshot_download(source)
        state = {}
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".safetensors"):
                    with safe_open(os.path.join(root, f), framework="pt") as sf:
                        for k in sf.keys():
                            if k.startswith("visual."):
                                state[k[len("visual."):]] = sf.get_tensor(k)

        own = self.char_vit.state_dict()
        loaded, skipped = [], []
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v.to(own[k].dtype)
                loaded.append(k)
            else:
                skipped.append(k)
        self.char_vit.load_state_dict(own)
        print(f"[FontDoctor] char-ViT: loaded {len(loaded)} tensors, skipped {len(skipped)}")
        if skipped:
            print(f"[FontDoctor]   skipped (re-initialized): {sorted(skipped)[:6]} ...")
        return skipped

    def load_deepencoder_weights(self, source: str) -> None:
        """Load official DeepSeek-OCR DeepEncoder weights (frozen branch)."""
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
        import os

        path = source if os.path.isdir(source) else snapshot_download(source)
        sam_state, clip_state = {}, {}
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".safetensors"):
                    with safe_open(os.path.join(root, f), framework="pt") as sf:
                        for k in sf.keys():
                            if k.startswith("sam_model."):
                                sam_state[k[len("sam_model."):]] = sf.get_tensor(k)
                            elif k.startswith("vision_model."):
                                clip_state[k[len("vision_model."):]] = sf.get_tensor(k)
        m1 = self.ocr_encoder.sam_model.load_state_dict(sam_state, strict=False)
        m2 = self.ocr_encoder.vision_model.load_state_dict(clip_state, strict=False)
        print(f"[FontDoctor] DeepEncoder SAM: {len(m1.missing_keys)} missing / {len(m1.unexpected_keys)} unexpected")
        print(f"[FontDoctor] DeepEncoder CLIP: {len(m2.missing_keys)} missing / {len(m2.unexpected_keys)} unexpected")
        # keep frozen
        for p in self.ocr_encoder.sam_model.parameters():
            p.requires_grad_(False)
        for p in self.ocr_encoder.vision_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # special tokens
    # ------------------------------------------------------------------ #
    def special_token_ids(self, tokenizer) -> Dict[str, int]:
        return {
            "glyph_img": tokenizer.convert_tokens_to_ids(GLYPH_IMG_TOKEN),
            "svg_img": tokenizer.convert_tokens_to_ids(SVG_IMG_TOKEN),
            "coord": tokenizer.convert_tokens_to_ids(COORD_TOKEN),
        }

    # ------------------------------------------------------------------ #
    # DeepStack injection (paper Eq. 8)
    # ------------------------------------------------------------------ #
    def _register_deepstack_hooks(self) -> None:
        layers = self.language_model.model.layers
        for k, layer_idx in enumerate(self.config.deepstack_decoder_layers):
            if layer_idx >= len(layers):
                raise ValueError(f"decoder has only {len(layers)} layers")
            hook = layers[layer_idx].register_forward_pre_hook(self._make_injection_hook(layer_idx))
            self._injection_hooks.append(hook)

    def _make_injection_hook(self, layer_idx: int):
        def hook(module, args):
            pending = self._pending_injections.get(layer_idx)
            if pending is None:
                return None
            hidden_states = args[0]
            positions, features = pending  # (N_vis,), (N_vis, D)
            if hidden_states.shape[1] <= 1:
                # incremental decoding step: injection only applies at prefill
                return None
            updated = hidden_states.clone()
            batch_idx, seq_idx = positions[:, 0], positions[:, 1]
            updated[batch_idx, seq_idx, :] = updated[batch_idx, seq_idx, :] + features.to(updated.dtype)
            return (updated,) + args[1:]

        return hook

    # ------------------------------------------------------------------ #
    # visual encoding
    # ------------------------------------------------------------------ #
    def encode_glyph_images(
        self, glyph_patches: torch.Tensor, glyph_grid_thw: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.char_vit(glyph_patches, glyph_grid_thw)

    def encode_svg_images(self, svg_prepared: List[dict]) -> List[torch.Tensor]:
        """Encode rendered SVG code images into Z_svg (E_ocr + g_ocr)."""
        return [self.ocr_encoder(p) for p in svg_prepared]

    # ------------------------------------------------------------------ #
    # embedding assembly
    # ------------------------------------------------------------------ #
    def _scatter_visual_embeddings(
        self,
        input_ids: torch.LongTensor,
        glyph_features: torch.Tensor,
        svg_features: List[torch.Tensor],
        ids: Dict[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace placeholder-token embeddings with visual embeddings.

        Returns the joint ``inputs_embeds`` and the ``(N_glyph, 2)`` position
        index (batch, seq) of glyph visual tokens -- the injection positions
        I_vis of paper Eq. 8.
        """
        embed_tokens = self.language_model.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids).clone()

        # glyph visual tokens (all images concatenated in batch order)
        glyph_mask = (input_ids == ids["glyph_img"])  # (B, T)
        num_glyph = int(glyph_mask.sum())
        if num_glyph != glyph_features.shape[0]:
            raise ValueError(
                f"<glyph_img> placeholders ({num_glyph}) != glyph features ({glyph_features.shape[0]})"
            )
        inputs_embeds[glyph_mask] = glyph_features.to(inputs_embeds.dtype)
        glyph_positions = glyph_mask.nonzero(as_tuple=False)  # (N, 2)

        # SVG optical tokens (all code images concatenated in batch order)
        svg_mask = input_ids == ids["svg_img"]
        num_svg = int(svg_mask.sum())
        if svg_features:
            all_svg = torch.cat(svg_features, dim=0)
        else:
            all_svg = torch.zeros(0, inputs_embeds.shape[-1], device=inputs_embeds.device)
        if num_svg != all_svg.shape[0]:
            raise ValueError(f"<svg_img> placeholders ({num_svg}) != Z_svg tokens ({all_svg.shape[0]})")
        if num_svg > 0:
            inputs_embeds[svg_mask] = all_svg.to(inputs_embeds.dtype)
        return inputs_embeds, glyph_positions

    def _split_deepstack_features(
        self, deepstack: List[torch.Tensor], glyph_positions: torch.Tensor
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        return {
            layer_idx: (glyph_positions, feats)
            for layer_idx, feats in zip(self.config.deepstack_decoder_layers, deepstack)
        }

    # ------------------------------------------------------------------ #
    # shared embedding assembly (training forward + inference prefill)
    # ------------------------------------------------------------------ #
    def assemble(
        self,
        input_ids: torch.LongTensor,
        glyph_patches: Optional[torch.Tensor],
        glyph_grid_thw: Optional[torch.Tensor],
        svg_prepared: Optional[List[dict]],
        tokenizer,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Encode all visual inputs and splice them into the text embeddings.

        Returns the joint ``inputs_embeds`` and the armed DeepStack injection
        dict ``{decoder_layer_idx: (positions, features)}``.
        """
        ids = self.special_token_ids(tokenizer)

        glyph_out = (
            self.encode_glyph_images(glyph_patches, glyph_grid_thw)
            if glyph_patches is not None
            else {"last": torch.zeros(0, self.config.language.hidden_size, device=input_ids.device),
                  "deepstack": []}
        )
        svg_feats = self.encode_svg_images(svg_prepared) if svg_prepared is not None else []

        inputs_embeds, glyph_positions = self._scatter_visual_embeddings(
            input_ids, glyph_out["last"], svg_feats, ids
        )

        pending: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        if glyph_positions.numel() > 0 and len(glyph_out["deepstack"]) > 0:
            pending = self._split_deepstack_features(glyph_out["deepstack"], glyph_positions)
        return inputs_embeds, pending

    # ------------------------------------------------------------------ #
    # forward (training)  --  L = L_tok + lambda_coord * L_coord
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        glyph_patches: Optional[torch.Tensor] = None,
        glyph_grid_thw: Optional[torch.Tensor] = None,
        svg_prepared: Optional[List[dict]] = None,
        coord_targets: Optional[torch.Tensor] = None,
        tokenizer=None,
        **lm_kwargs,
    ) -> Tuple[CausalLMOutputWithPast, Dict[str, Optional[torch.Tensor]]]:
        ids = self.special_token_ids(tokenizer)

        inputs_embeds, pending = self.assemble(
            input_ids, glyph_patches, glyph_grid_thw, svg_prepared, tokenizer
        )
        self._pending_injections = pending
        try:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=None,
                output_hidden_states=True,
                **lm_kwargs,
            )
        finally:
            self._pending_injections = {}

        hidden = outputs.hidden_states[-1]  # (B, T, D) final layer
        logits = self.language_model.lm_head(hidden)

        loss = None
        tok_loss = None
        coord_loss = None
        if labels is not None:
            # standard next-token cross entropy (L_tok)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            tok_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # continuous coordinate regression (L_coord, paper Eq. 13)
            if coord_targets is not None:
                coord_mask = labels == ids["coord"]  # placeholder positions
                if int(coord_mask.sum()) != coord_targets.shape[0]:
                    raise ValueError(
                        f"<coord> tokens ({int(coord_mask.sum())}) != coord targets ({coord_targets.shape[0]})"
                    )
                coord_hidden = hidden[coord_mask]  # (M, D)
                pred_coords = self.coord_head(coord_hidden)
                coord_loss = F.l1_loss(pred_coords, coord_targets.to(pred_coords.dtype))

            loss = tok_loss
            if coord_loss is not None:
                loss = loss + self.config.lambda_coord * coord_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), {"tok_loss": tok_loss, "coord_loss": coord_loss}

    # ------------------------------------------------------------------ #
    # stage-wise parameter freezing (paper Algorithm 2)
    # ------------------------------------------------------------------ #
    def set_stage(self, stage: int) -> None:
        """Configure trainable parameter subsets for the 3-stage protocol.

        Stage I   (SOR): train g_ocr projector + LM interface;
                         E_ocr frozen, E_v frozen, coord head frozen.
        Stage II  (RVG): train E_v (+mergers) + LM; E_ocr branch frozen.
        Stage III (DET): train everything except the frozen E_ocr tower.
        """
        def _set(module: nn.Module, flag: bool):
            for p in module.parameters():
                p.requires_grad_(flag)

        def _set_ocr_interface(flag: bool):
            # g_ocr projector + the structural newline / separator embeddings
            _set(self.ocr_encoder.projector, flag)
            self.ocr_encoder.image_newline.requires_grad_(flag)
            self.ocr_encoder.view_seperator.requires_grad_(flag)

        # E_ocr tower is frozen in ALL stages (paper Sec. 4.2).
        _set(self.ocr_encoder.sam_model, False)
        _set(self.ocr_encoder.vision_model, False)

        if stage == 1:
            _set(self.char_vit, False)
            _set(self.coord_head, False)
            _set_ocr_interface(True)
            _set(self.language_model, True)
        elif stage == 2:
            _set(self.char_vit, True)
            _set(self.coord_head, False)
            _set_ocr_interface(False)
            _set(self.language_model, True)
        elif stage == 3:
            _set(self.char_vit, True)
            _set(self.coord_head, True)
            _set_ocr_interface(True)
            _set(self.language_model, True)
        else:
            raise ValueError(f"unknown stage {stage}")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[FontDoctor] stage {stage}: {trainable/1e9:.2f}B / {total/1e9:.2f}B trainable params")
