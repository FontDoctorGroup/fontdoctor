"""Single-sample FontDoctor inference CLI.

Referenced mode:
    python scripts/infer.py --checkpoint ckpt/fontdoctor.pt \
        --image qry.png --svg qry.svg \
        --ref-images ref1.png ref2.png --ref-svgs ref1.svg ref2.svg \
        --out pred.svg

Universal mode (no references):
    python scripts/infer.py --checkpoint ckpt/fontdoctor.pt --image qry.png --svg qry.svg --out pred.svg
"""

import argparse
from pathlib import Path

from PIL import Image

import common
from fontdoctor.configuration_fontdoctor import FontDoctorConfig
from fontdoctor.inference.pipeline import FontDoctorInference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen3", default="Qwen/Qwen3-4B")
    parser.add_argument("--image", required=True, help="query glyph raster image")
    parser.add_argument("--svg", required=True, help="query glyph SVG XML file")
    parser.add_argument("--ref-images", nargs="*", default=[])
    parser.add_argument("--ref-svgs", nargs="*", default=[])
    parser.add_argument("--out", default="pred.svg")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()

    if args.ref_images and len(args.ref_images) != len(args.ref_svgs):
        raise ValueError("--ref-images and --ref-svgs must have the same length")

    config = FontDoctorConfig()
    config.language.model_name_or_path = args.qwen3
    tokenizer = common.build_tokenizer(args.qwen3)
    model = common.build_model(config, tokenizer, checkpoint=args.checkpoint)
    pipeline = FontDoctorInference(model, tokenizer, config)

    query_image = Image.open(args.image).convert("RGB")
    query_svg = Path(args.svg).read_text(encoding="utf-8")
    reference_pairs = [
        (Image.open(p).convert("RGB"), Path(s).read_text(encoding="utf-8"))
        for p, s in zip(args.ref_images, args.ref_svgs)
    ] or None

    pred_svg = pipeline.detect(
        query_image, query_svg,
        reference_pairs=reference_pairs,
        max_new_tokens=args.max_new_tokens,
    )
    Path(args.out).write_text(pred_svg, encoding="utf-8")
    print(f"prediction written to {args.out}")


if __name__ == "__main__":
    main()
