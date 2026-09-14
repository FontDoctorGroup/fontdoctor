"""Stage I: Optical SVG Branch Calibration (SOR) -- paper Algorithm 2.

Train g_ocr and the language-model interface so that the model recovers the
original SVG XML text S from its rendered code image:  max p(S | Z_svg),
with E_ocr frozen.

Example:
    python scripts/train_stage1_sor.py \
        --manifest data/cfdefect/manifest_train.jsonl \
        --image-root data/cfdefect \
        --qwen3 Qwen/Qwen3-4B \
        --deepseek-ocr deepseek-ai/DeepSeek-OCR
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
    parser.add_argument("--deepseek-ocr", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--output-dir", default="runs/stage1_sor")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = FontDoctorConfig()
    config.language.model_name_or_path = args.qwen3

    tokenizer = common.build_tokenizer(args.qwen3)
    model = common.build_model(config, tokenizer, deepseek_ocr_path=args.deepseek_ocr)

    dataset = CFDefect2MDataset(args.manifest, args.image_root, config, stage=1)
    train_args = TrainArgs(
        output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, num_workers=args.num_workers,
    )
    FontDoctorTrainer(model, tokenizer, config, train_args, stage=1).train(dataset)


if __name__ == "__main__":
    main()
