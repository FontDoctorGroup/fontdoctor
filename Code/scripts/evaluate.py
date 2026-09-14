"""Evaluate FontDoctor on CFDefect-2M test splits (paper Sec. 4).

Computes the dual-domain metrics on the test manifest:
    MSE / IoU / Precision / Recall / F1   (raster defect maps)
    DTW / LDTW / CD                       (vector defect contours)

Example:
    python scripts/evaluate.py \
        --checkpoint runs/stage3_det/stage3-final/fontdoctor.pt \
        --manifest data/cfdefect/manifest_test.jsonl \
        --image-root data/cfdefect \
        --mode referenced --num-refs 6 --max-samples 500
"""

import argparse
import json
import random

import torch
from PIL import Image

import common
from fontdoctor.configuration_fontdoctor import FontDoctorConfig
from fontdoctor.data.dataset import load_manifest
from fontdoctor.eval.metrics import MetricAverager, evaluate_prediction
from fontdoctor.inference.pipeline import FontDoctorInference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--qwen3", default="Qwen/Qwen3-4B")
    parser.add_argument("--mode", choices=["referenced", "universal"], default="referenced")
    parser.add_argument("--num-refs", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-preds", default=None, help="optional JSONL of predictions")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = FontDoctorConfig()
    config.language.model_name_or_path = args.qwen3
    tokenizer = common.build_tokenizer(args.qwen3)
    model = common.build_model(config, tokenizer, checkpoint=args.checkpoint)
    pipeline = FontDoctorInference(model, tokenizer, config)

    records = load_manifest(args.manifest)
    defective = [r for r in records if r.has_defect and r.target_svg]
    clean_by_font = {}
    for i, r in enumerate(records):
        if not r.has_defect:
            clean_by_font.setdefault(r.font_id, []).append(i)
    if args.max_samples > 0:
        defective = defective[: args.max_samples]

    averager = MetricAverager()
    pred_file = open(args.save_preds, "w", encoding="utf-8") if args.save_preds else None

    for n, rec in enumerate(defective):
        query_image = Image.open(f"{args.image_root}/{rec.image_path}").convert("RGB")

        reference_pairs = None
        if args.mode == "referenced":
            pool = clean_by_font.get(rec.font_id, [])
            idxs = random.sample(pool, k=min(args.num_refs, len(pool)))
            reference_pairs = [
                (Image.open(f"{args.image_root}/{records[j].image_path}").convert("RGB"),
                 records[j].svg)
                for j in idxs
            ]

        pred_svg = pipeline.detect(
            query_image, rec.svg,
            reference_pairs=reference_pairs,
            max_new_tokens=args.max_new_tokens,
        )
        metrics = evaluate_prediction(pred_svg, rec.target_svg)
        averager.update(metrics)

        if pred_file:
            pred_file.write(json.dumps({"id": rec.id, "pred": pred_svg,
                                        "target": rec.target_svg, **metrics}) + "\n")
        if (n + 1) % 10 == 0:
            print(f"[{n + 1}/{len(defective)}] running: {averager.compute()}")

    if pred_file:
        pred_file.close()

    print("\n==== Final results ====")
    for k, v in averager.compute().items():
        print(f"{k:>10s}: {v:.4f}")


if __name__ == "__main__":
    main()
