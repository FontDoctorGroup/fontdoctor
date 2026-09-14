"""Stage II: Character Raster-to-Vector Alignment (RVG) -- paper Algorithm 2.

Train the character vision encoder E_v (+ mergers) and the language model to
generate the SVG XML code S of a real raster glyph x_v:
    max p(S | E_v(x_v)),  with the E_ocr branch untouched.

Example:
    python scripts/train_stage2_rvg.py \
        --manifest data/cfdefect/manifest_train.jsonl \
        --image-root data/cfdefect \
        --qwen3 Qwen/Qwen3-4B \
        --char-vit Qwen/Qwen2.5-VL-3B-Instruct \
        --init-stage1 runs/stage1_sor/stage1-final/fontdoctor.pt
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
    parser.add_argument("--char-vit", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="official Qwen2.5-VL checkpoint (vision tower init)")
    parser.add_argument("--deepseek-ocr", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--init-stage1", default=None, help="Stage-I checkpoint (optional)")
    parser.add_argument("--output-dir", default="runs/stage2_rvg")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    config = FontDoctorConfig()
    config.language.model_name_or_path = args.qwen3

    tokenizer = common.build_tokenizer(args.qwen3)
    model = common.build_model(
        config, tokenizer,
        char_vit_path=args.char_vit,
        deepseek_ocr_path=args.deepseek_ocr,
        checkpoint=args.init_stage1,
    )

    dataset = CFDefect2MDataset(args.manifest, args.image_root, config, stage=2)
    train_args = TrainArgs(
        output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, num_workers=args.num_workers,
    )
    FontDoctorTrainer(model, tokenizer, config, train_args, stage=2).train(dataset)


if __name__ == "__main__":
    main()
