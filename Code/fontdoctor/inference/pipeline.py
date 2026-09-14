"""FontDoctor inference pipelines.

Two complementary detection paradigms (paper Sec. 3.2):

* **Few-Shot Template-Referenced** -- the query glyph is conditioned on K
  defect-free same-style reference pairs (I_ref, S_ref);
* **Non-Referenced Universal** -- detection from (I_qry, S_qry) alone, using
  the universal defect patterns learned during training.

Decoding implements the CCED inference procedure (paper Sec. 3.6): the model
first autoregressively generates the SVG *structural skeleton* (with
``<coord>`` placeholders); the hidden state of every placeholder is then fed
to the coordinate head g_phi and the regressed coordinates are filled back
into the skeleton, yielding a syntactically valid, precisely localized SVG
defect annotation.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image

from ..configuration_fontdoctor import FontDoctorConfig
from ..constants import COORD_TOKEN, EOV_TOKEN, GLYPH_IMG_TOKEN, SVG_IMG_TOKEN
from ..data.collator import FontDoctorCollator, build_chat_prompt
from ..data.dataset import SYSTEM_REFERENCED, SYSTEM_UNIVERSAL, USER_REFERENCED, USER_UNIVERSAL
from ..data.svg_renderer import render_svg_text_image
from ..data.svg_utils import skeleton_to_svg


class FontDoctorInference:
    """End-to-end defect-detection inference wrapper."""

    def __init__(self, model, tokenizer, config: FontDoctorConfig, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.collator = FontDoctorCollator(config, tokenizer, ocr_encoder=self.model.ocr_encoder)

        self.coord_id = tokenizer.convert_tokens_to_ids(COORD_TOKEN)
        self.eov_id = tokenizer.convert_tokens_to_ids(EOV_TOKEN)
        self.eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # ------------------------------------------------------------------ #
    # input preparation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _prepare(self, sample: dict) -> dict:
        """Reuse the collator to build a batch-of-one prompt (generation mode)."""
        return self.collator([{**sample, "target": None, "coord_targets": None}])

    # ------------------------------------------------------------------ #
    # CCED greedy decoding
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _generate_skeleton(
        self, batch: dict, max_new_tokens: int = 8192
    ) -> Tuple[str, torch.Tensor]:
        """Greedy autoregressive skeleton generation.

        Returns the decoded skeleton text and the ``(M, D)`` hidden states of
        the generated ``<coord>`` placeholders.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        inputs_embeds, pending = self.model.assemble(
            input_ids,
            batch["glyph_patches"].to(self.device) if batch["glyph_patches"] is not None else None,
            batch["glyph_grid_thw"].to(self.device) if batch["glyph_grid_thw"] is not None else None,
            batch["svg_prepared"],
            self.tokenizer,
        )

        embed_tokens = self.model.language_model.get_input_embeddings()
        coord_hiddens: List[torch.Tensor] = []
        generated: List[int] = []

        # ---- prefill ----
        self.model._pending_injections = pending
        try:
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
            )
        finally:
            self.model._pending_injections = {}

        past = outputs.past_key_values

        for _ in range(max_new_tokens):
            next_logits = outputs.logits[:, -1, :]
            next_id = int(torch.argmax(next_logits, dim=-1).item())

            if next_id in (self.eov_id, self.eos_id, self.tokenizer.eos_token_id):
                break

            generated.append(next_id)
            is_coord = next_id == self.coord_id

            # ---- incremental step ----
            step_embed = embed_tokens(torch.tensor([[next_id]], device=self.device))
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)], dim=1
            )
            outputs = self.model.language_model(
                inputs_embeds=step_embed,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values

            if is_coord:
                # hidden state AFTER consuming the <coord> placeholder, which
                # is exactly the position supervised by L_coord in training
                coord_hiddens.append(outputs.hidden_states[-1][0, -1].detach())

        skeleton_text = self.tokenizer.decode(generated, skip_special_tokens=False)
        coord_hidden = (
            torch.stack(coord_hiddens, dim=0)
            if coord_hiddens
            else torch.zeros(0, self.config.language.hidden_size, device=self.device)
        )
        return skeleton_text, coord_hidden

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def detect(
        self,
        query_image: Image.Image,
        query_svg: str,
        reference_pairs: Optional[Sequence[Tuple[Image.Image, str]]] = None,
        max_new_tokens: int = 8192,
    ) -> str:
        """Run defect detection and return the defect-annotated SVG.

        Args:
            query_image     : raster image I_qry of the glyph under test.
            query_svg       : SVG XML string S_qry of the glyph under test.
            reference_pairs : optional K (I_ref, S_ref) defect-free same-style
                pairs.  ``None`` / empty selects the Non-Referenced Universal
                paradigm; otherwise Few-Shot Template-Referenced detection.
            max_new_tokens  : decoding budget of the skeleton.
        """
        referenced = bool(reference_pairs)
        ref_images = [img for img, _ in (reference_pairs or [])]
        ref_svg_images = [
            render_svg_text_image(
                svg,
                max_width=self.config.svg_render_max_width,
                font_size=self.config.svg_render_font_size,
                line_spacing=self.config.svg_render_line_spacing,
            )
            for _, svg in (reference_pairs or [])
        ]

        sample = {
            "system": SYSTEM_REFERENCED if referenced else SYSTEM_UNIVERSAL,
            "user": USER_REFERENCED if referenced else USER_UNIVERSAL,
            "query_image": query_image,
            "query_svg_image": render_svg_text_image(
                query_svg,
                max_width=self.config.svg_render_max_width,
                font_size=self.config.svg_render_font_size,
                line_spacing=self.config.svg_render_line_spacing,
            ),
            "ref_images": ref_images,
            "ref_svg_images": ref_svg_images,
        }

        batch = self._prepare(sample)
        skeleton, coord_hidden = self._generate_skeleton(batch, max_new_tokens)

        # ---- coordinate regression + fill-back (CCED inference) ----------
        if coord_hidden.shape[0] > 0:
            pred = self.model.coord_head(coord_hidden.to(self.device))
            coords = pred.float().cpu().numpy() * float(self.config.coord_normalize)
            coords = coords.clip(0, self.config.coord_normalize)
            coord_list = [(float(x), float(y)) for x, y in coords]
        else:
            coord_list = []

        # strip the SOV marker and keep the document body
        body = skeleton
        for marker in ("<sov>", "<eov>"):
            body = body.replace(marker, "")
        body = body.strip()
        return skeleton_to_svg(body, coord_list)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def vectorize(self, query_image: Image.Image, max_new_tokens: int = 8192) -> str:
        """Stage-II capability: raster glyph -> SVG XML (Raster-to-Vector)."""
        from ..data.dataset import SYSTEM_RVG, USER_RVG

        sample = {
            "system": SYSTEM_RVG,
            "user": USER_RVG,
            "query_image": query_image,
            "query_svg_image": None,
            "ref_images": [],
            "ref_svg_images": [],
        }
        batch = self._prepare(sample)
        skeleton, coord_hidden = self._generate_skeleton(batch, max_new_tokens)
        if coord_hidden.shape[0] > 0:
            pred = self.model.coord_head(coord_hidden.to(self.device))
            coords = (pred.float().cpu().numpy() * float(self.config.coord_normalize)).clip(
                0, self.config.coord_normalize
            )
            coord_list = [(float(x), float(y)) for x, y in coords]
        else:
            coord_list = []
        body = skeleton.replace("<sov>", "").replace("<eov>", "").strip()
        return skeleton_to_svg(body, coord_list)

    @torch.no_grad()
    def read_svg(self, svg_code_image: Image.Image, max_new_tokens: int = 8192) -> str:
        """Stage-I capability: optical reading of a rendered SVG code image."""
        from ..data.dataset import SYSTEM_SOR, USER_SOR

        sample = {
            "system": SYSTEM_SOR,
            "user": USER_SOR,
            "query_image": None,
            "query_svg_image": svg_code_image,
            "ref_images": [],
            "ref_svg_images": [],
        }
        batch = self._prepare(sample)
        text, _ = self._generate_skeleton(batch, max_new_tokens)
        return text.replace("<sov>", "").replace("<eov>", "").strip()
