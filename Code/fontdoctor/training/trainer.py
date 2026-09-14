"""Three-stage FontDoctor training protocol (paper Algorithm 2).

Stage I   -- Optical SVG Branch Calibration (SOR):
             train g_ocr + LM interface with L_SOR = -log p(S | Z_svg),
             E_ocr frozen.
Stage II  -- Character Raster-to-Vector Alignment (RVG):
             train E_v + mergers + LM with L_RVG = -log p(S | E_v(x_v)),
             E_ocr branch frozen.
Stage III -- Curriculum-based defect detection:
             stroke-count curriculum + Bernoulli task mixing +
             L_det = L_tok + lambda_coord * L_coord.
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..configuration_fontdoctor import FontDoctorConfig
from ..data.collator import FontDoctorCollator
from ..data.dataset import CFDefect2MDataset, StrokeCountCurriculumSampler


@dataclass
class TrainArgs:
    output_dir: str = "runs/fontdoctor"
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    num_workers: int = 4
    log_every: int = 20
    save_every: int = 1000
    seed: int = 42
    resume: Optional[str] = None
    chunk_shuffle: int = 256  # within-curriculum-window shuffle (0 = strict)


class FontDoctorTrainer:
    """Minimal bf16 trainer driving one training stage."""

    def __init__(
        self,
        model,
        tokenizer,
        config: FontDoctorConfig,
        args: TrainArgs,
        stage: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.stage = stage

        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.set_stage(stage)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
        )
        self.collator = FontDoctorCollator(config, tokenizer, ocr_encoder=self.model.ocr_encoder)

    # ------------------------------------------------------------------ #
    def _build_loader(self, dataset: CFDefect2MDataset) -> DataLoader:
        if self.stage == 3:
            # stroke-count curriculum: sort D by stroke_count (Algorithm 2)
            sampler = StrokeCountCurriculumSampler(dataset, chunk_shuffle=self.args.chunk_shuffle)
            return DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                sampler=sampler,
                num_workers=self.args.num_workers,
                collate_fn=self.collator,
                pin_memory=True,
                drop_last=True,
            )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
        )

    def _lr_lambda(self, step: int, total: int) -> float:
        warmup = max(1, int(total * self.args.warmup_ratio))
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # ------------------------------------------------------------------ #
    def train(self, dataset: CFDefect2MDataset) -> None:
        args = self.args
        loader = self._build_loader(dataset)
        steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
        total_steps = steps_per_epoch * args.epochs
        schedule = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: self._lr_lambda(s, total_steps)
        )

        os.makedirs(args.output_dir, exist_ok=True)
        global_step = 0
        start = time.time()

        for epoch in range(args.epochs):
            self.model.train()
            # keep the frozen OCR tower in eval mode (no dropout/BN drift)
            self.model.ocr_encoder.sam_model.eval()
            self.model.ocr_encoder.vision_model.eval()

            for it, batch in enumerate(loader):
                batch = self._to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda"):
                    outputs, aux = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        glyph_patches=batch["glyph_patches"],
                        glyph_grid_thw=batch["glyph_grid_thw"],
                        svg_prepared=batch["svg_prepared"],
                        coord_targets=batch["coord_targets"],
                        tokenizer=self.tokenizer,
                    )
                    loss = outputs.loss / args.grad_accum

                loss.backward()

                if (it + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                    self.optimizer.step()
                    schedule.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.log_every == 0:
                        tok = float(aux["tok_loss"].detach()) if aux["tok_loss"] is not None else -1
                        coord = (
                            float(aux["coord_loss"].detach()) if aux["coord_loss"] is not None else -1
                        )
                        lr = schedule.get_last_lr()[0]
                        elapsed = time.time() - start
                        print(
                            f"[stage {self.stage}] epoch {epoch} step {global_step}/{total_steps} "
                            f"loss {tok:.4f} coord {coord:.4f} lr {lr:.2e} "
                            f"({elapsed/60:.1f} min)"
                        )

                    if global_step % args.save_every == 0:
                        self.save(os.path.join(args.output_dir, f"stage{self.stage}-step{global_step}"))

        self.save(os.path.join(args.output_dir, f"stage{self.stage}-final"))

    # ------------------------------------------------------------------ #
    def _to_device(self, batch: dict) -> dict:
        out = dict(batch)
        for key in ("input_ids", "attention_mask", "labels", "glyph_patches",
                    "glyph_grid_thw", "coord_targets"):
            if out.get(key) is not None:
                out[key] = out[key].to(self.device)
        # svg_prepared holds CPU PIL-derived tensors; the OCR branch moves
        # them onto its own device inside `encode`.
        return out

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # save only trainable-related modules (the LM is saved fully for
        # standalone inference), plus config + tokenizer for reproducibility.
        torch.save(self.model.state_dict(), os.path.join(path, "fontdoctor.pt"))
        with open(os.path.join(path, "train_args.json"), "w") as f:
            json.dump(vars(self.args), f, indent=2)
        self.tokenizer.save_pretrained(path)
        print(f"[stage {self.stage}] checkpoint saved to {path}")
