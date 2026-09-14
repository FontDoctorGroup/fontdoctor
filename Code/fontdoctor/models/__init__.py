"""FontDoctor model zoo.

Submodules are imported lazily (PEP 562) so that lightweight users (e.g. the
SVG data utilities) do not pay for / require the heavy backbone deps.
"""

import importlib

__all__ = [
    "CharacterVisionEncoder",
    "OpticalCompressionEncoder",
    "CoordinateRegressionHead",
    "FontDoctorModel",
    "image_to_patches",
    "resize_to_multiple_of_patch",
    "dynamic_preprocess",
]

_LAZY = {
    "CharacterVisionEncoder": ".character_vit",
    "image_to_patches": ".character_vit",
    "resize_to_multiple_of_patch": ".character_vit",
    "OpticalCompressionEncoder": ".ocr_branch",
    "dynamic_preprocess": ".ocr_branch",
    "CoordinateRegressionHead": ".projectors",
    "FontDoctorModel": ".modeling_fontdoctor",
}


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
