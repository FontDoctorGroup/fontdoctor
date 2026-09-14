"""Trainable projection / regression heads of FontDoctor."""

import torch
import torch.nn as nn


class CoordinateRegressionHead(nn.Module):
    """Continuous coordinate head g_phi of the CCED (paper Sec. 3.6, Eq. 12).

    A two-layer MLP maps the hidden state h_m of every ``<coord>`` placeholder
    to a 2-D coordinate::

        c_hat_m = W2 * sigma(W1 * h_m + b1) + b2,   c_hat_m in R^2

    Coordinates are regressed in the normalized [0, 1] range (divide by the
    SVG canvas size, typically 512); de-normalization happens in inference.
    """

    def __init__(self, hidden_size: int, mlp_ratio: int = 1):
        super().__init__()
        inner = hidden_size * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, 2),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (..., hidden_size) -> (..., 2) normalized coords."""
        return self.mlp(hidden_states)
