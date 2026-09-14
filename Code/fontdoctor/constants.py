"""Special tokens and task constants used across FontDoctor.

The tokenizer is the official Qwen3 tokenizer extended with a small set of
task-specific special tokens (added via ``tokenizer.add_special_tokens`` and
``model.resize_token_embeddings``):

* ``<glyph_img>``  : placeholder replaced by character-ViT embeddings of the
  raster glyph image x_v (possibly several merged visual tokens).
* ``<svg_img>``    : placeholder replaced by optical-compression embeddings
  Z_svg of one rendered SVG XML code image.
* ``<sov>`` / ``<eov>`` : start / end of the vector (SVG) target sequence,
  mirroring the SOV / EOV markers in paper Table (Detection Paradigms).
* ``<coord>``      : placeholder that substitutes every 2-D coordinate pair of
  the target SVG skeleton; its hidden state is fed to the continuous
  coordinate regression head (paper Sec. 3.6, CCED).
"""

GLYPH_IMG_TOKEN = "<glyph_img>"
SVG_IMG_TOKEN = "<svg_img>"
SOV_TOKEN = "<sov>"
EOV_TOKEN = "<eov>"
COORD_TOKEN = "<coord>"

EXTRA_SPECIAL_TOKENS = [
    GLYPH_IMG_TOKEN,
    SVG_IMG_TOKEN,
    SOV_TOKEN,
    EOV_TOKEN,
    COORD_TOKEN,
]

# SVG path drawing commands kept by FontDoctor (paper Appendix, 7 commands).
SVG_COMMANDS = ["M", "L", "H", "V", "Q", "C", "Z"]

# Number of 2-D coordinate pairs consumed by each command.  During
# preprocessing H / V are normalized to L, so every kept command is absolute.
COMMAND_ARITY = {
    "M": 1,
    "L": 1,
    "H": 1,   # normalized away before modeling, kept for parsing robustness
    "V": 1,   # normalized away before modeling, kept for parsing robustness
    "Q": 2,
    "C": 3,
    "Z": 0,
}

# Canonical canvas of the serialized glyph SVG (paper: typical 512 x 512).
CANVAS_SIZE = 512

# Fill color of the expert defect-annotation groups.
DEFECT_FILL_RGB = (255, 0, 0)
DEFECT_GROUP_CLASS = "defect"
