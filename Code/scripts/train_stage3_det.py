"""Stage III: Curriculum-based defect detection (DET) -- paper Algorithm 2.

Fine-tune the defect decoder on S_tgt = S (+) A with
    L_det = L_tok + lambda_coord * L_coord,
using the stroke-count curriculum and Bernoulli mixing of the
template-referenced / non-referenced tasks (alpha ~ B(lambda)).

Example:
    python scripts/train_stage3_det.py \
        --manifest data/cfdefect/manifest_train.jsonl \
        --image-root data/cfdefect \
        --qwen3 Qwen/Qwen3-4B \
        --init-stage2 runs/stage2_rvg/stage2-final/fontdoctor.pt \
        --lambda-coord 1.0 --referenced-prob 0.5 --max-refs 6
"""

import argparse

import common
from fontdoctor.configuration_fontdoctor import FontDoctorConfig
from fontdoctor.data.dataset import CFDefect2MDataset
from fontdoctor.training.trainer import FontDoctorTrainer, TrainArgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--qwen3", default="Qwen/Qwen3-4B")
    parser.add_argument("--char-vit", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--deepseek-ocr", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--init-stage2", default=None, help="Stage-II checkpoint (recommended)")
    parser.add_argument("--output-dir", default="runs/stage3_det")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lambda-coord", type=float, default=1.0)
    parser.add_argument("--referenced-prob", type=float, default=0.5)
    parser.add_argument("--max-refs", type=int, default=6)
    args = parser.parse_args()

    config = FontDoctorConfig()
    config.language.model_name_or_path = args.qwen3
    config.lambda_coord = args.lambda_coord
    config.referenced_mode_prob = args.referenced_prob
    config.max_reference_templates = args.max_refs

    tokenizer = common.build_tokenizer(args.qwen3)
    model = common.build_model(
        config, tokenizer,
        char_vit_path=args.char_vit,
        deepseek_ocr_path=args.deepseek_ocr,
        checkpoint=args.init_stage2,
    )

    dataset = CFDefect2MDataset(args.manifest, args.image_root, config, stage=3)
    train_args = TrainArgs(
        output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, num_workers=args.num_workers,
    )
    FontDoctorTrainer(model, tokenizer, config, train_args, stage=3).train(dataset)


if __name__ == "__main__":
    main()
