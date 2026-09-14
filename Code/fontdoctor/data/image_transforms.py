"""Minimal image transforms (ToTensor + Normalize) without torchvision.

Keeps FontDoctor's image preprocessing dependency-light while matching the
exact numerics of ``torchvision.transforms.ToTensor`` followed by
``Normalize(mean, std)``.
"""

from typing import Sequence

import numpy as np
import torch
from PIL import Image


class SimpleImageTransform:
    """PIL image -> normalized CHW float tensor.

    Equivalent to::

        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0  # HWC, [0, 1]
        tensor = torch.from_numpy(array).permute(2, 0, 1)    # CHW
        return (tensor - self.mean) / self.std
