"""Dual-domain evaluation metrics of FontDoctor (paper Appendix
"Evaluation Metrics").

Raster domain:
    * MSE  between the rendered defect maps (paper Eq. 15)
    * IoU / Precision / Recall / F1 of the defect regions

Vector domain:
    * DTW   -- Dynamic Time Warping, average single-step cost (Eq. 16-17)
    * LDTW  -- Localized DTW with a Sakoe-Chiba window constraint (Eq. 18)
    * CD    -- symmetric Chamfer Distance of defect contour point sets
               (Eq. 19)

Optical-reading diagnostics (Stage I):
    * CER, BLEU-4, Syntax Validity, Exact Match
"""

import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.svg_renderer import rasterize_defect_map
from ..data.svg_utils import sample_svg_defect_points

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Raster-domain metrics
# ---------------------------------------------------------------------------

def defect_map_arrays(pred_svg: str, target_svg: str, size=(512, 512)) -> Tuple[np.ndarray, np.ndarray]:
    """Render both SVG defect annotations to float arrays in [0, 1]."""
    pred = np.asarray(rasterize_defect_map(pred_svg, size), dtype=np.float32) / 255.0
    target = np.asarray(rasterize_defect_map(target_svg, size), dtype=np.float32) / 255.0
    return pred, target


def mse_defect_maps(pred_svg: str, target_svg: str, size=(512, 512)) -> float:
    """Mean Squared Error of the rendered defect maps (paper Eq. 15)."""
    pred, target = defect_map_arrays(pred_svg, target_svg, size)
    return float(np.mean((pred - target) ** 2))


def iou_precision_recall_f1(
    pred_svg: str, target_svg: str, size=(512, 512), thresh: float = 0.5
) -> Dict[str, float]:
    """Pixel-level defect-region statistics on the binarized defect maps."""
    pred, target = defect_map_arrays(pred_svg, target_svg, size)
    p = pred > thresh
    t = target > thresh
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    inter = tp
    union = float(np.logical_or(p, t).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp + fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp + fp == 0 else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = inter / union if union > 0 else 1.0
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Vector-domain metrics
# ---------------------------------------------------------------------------

def _dtw_with_path_length(P: np.ndarray, Q: np.ndarray, window: Optional[int] = None) -> Tuple[float, int]:
    """DTW cumulative cost D(n, m) and the optimal warping-path length.

    ``window`` implements the local constraint C of LDTW (Sakoe-Chiba band);
    ``None`` gives unconstrained global DTW.
    """
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return 0.0, 1
    w = max(window or max(n, m), abs(n - m))

    INF = float("inf")
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    L = np.zeros((n + 1, m + 1), dtype=np.int64)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = math.hypot(P[i - 1][0] - Q[j - 1][0], P[i - 1][1] - Q[j - 1][1])
            choices = (D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            k = int(np.argmin(choices))
            D[i, j] = cost + choices[k]
            L[i, j] = (L[i - 1, j], L[i, j - 1], L[i - 1, j - 1])[k] + 1

    return float(D[n, m]), int(max(L[n, m], 1))


def dtw_average_step_cost(
    pred_points: Sequence[Point], target_points: Sequence[Point], window_ratio: Optional[float] = None
) -> float:
    """DTW / LDTW average single-step cost (paper Eq. 17 / 18).

    ``window_ratio=None``   -> global DTW;
    ``window_ratio=r``      -> LDTW with window r * max(n, m).
    """
    P = np.asarray(pred_points, dtype=np.float64).reshape(-1, 2)
    Q = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if len(P) == 0 or len(Q) == 0:
        return float("nan")
    window = None
    if window_ratio is not None:
        window = max(1, int(round(window_ratio * max(len(P), len(Q)))))
    cost, path_len = _dtw_with_path_length(P, Q, window=window)
    return cost / path_len


def chamfer_distance(
    pred_points: Sequence[Point], target_points: Sequence[Point], max_points: int = 4096
) -> float:
    """Symmetric Chamfer Distance of two point sets (paper Eq. 19)."""
    A = np.asarray(pred_points, dtype=np.float64).reshape(-1, 2)
    B = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    if len(A) > max_points:
        A = A[np.random.choice(len(A), max_points, replace=False)]
    if len(B) > max_points:
        B = B[np.random.choice(len(B), max_points, replace=False)]
    # pairwise squared distances, chunked to bound memory
    def _min_dist(X, Y):
        out = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), 1024):
            chunk = X[start : start + 1024]
            d2 = ((chunk[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
            out[start : start + 1024] = np.sqrt(d2.min(axis=1))
        return out

    return float(_min_dist(A, B).mean() + _min_dist(B, A).mean())


# ---------------------------------------------------------------------------
# Optical-reading diagnostics (Stage I)
# ---------------------------------------------------------------------------

def character_error_rate(pred: str, target: str) -> float:
    """Levenshtein distance normalized by the target length."""
    n, m = len(pred), len(target)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pred[i - 1] != target[j - 1]))
        prev = cur
    return prev[m] / m


def bleu4(pred: str, target: str) -> float:
    """Self-contained corpus BLEU-4 over whitespace tokens."""
    pred_toks, target_toks = pred.split(), target.split()
    if not pred_toks:
        return 0.0
    scores = []
    for n in range(1, 5):
        pred_ngrams: Dict[tuple, int] = {}
        for i in range(len(pred_toks) - n + 1):
            ng = tuple(pred_toks[i : i + n])
            pred_ngrams[ng] = pred_ngrams.get(ng, 0) + 1
        target_ngrams: Dict[tuple, int] = {}
        for i in range(len(target_toks) - n + 1):
            ng = tuple(target_toks[i : i + n])
            target_ngrams[ng] = target_ngrams.get(ng, 0) + 1
        clipped = sum(min(c, target_ngrams.get(ng, 0)) for ng, c in pred_ngrams.items())
        total = max(1, sum(pred_ngrams.values()))
        scores.append(clipped / total if total else 0.0)
    if min(scores) == 0:
        return 0.0
    bp = 1.0 if len(pred_toks) > len(target_toks) else math.exp(1 - len(target_toks) / max(1, len(pred_toks)))
    return bp * math.exp(sum(math.log(s) for s in scores) / 4)


def svg_syntax_valid(svg: str) -> bool:
    """Well-formed XML, contains a root <svg> element and path data."""
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return False
    if not root.tag.endswith("svg"):
        return False
    return any(el.tag.endswith("path") for el in root.iter())


# ---------------------------------------------------------------------------
# Joint evaluation
# ---------------------------------------------------------------------------

def evaluate_prediction(
    pred_svg: str,
    target_svg: str,
    size=(512, 512),
    ldtw_window_ratio: float = 0.1,
) -> Dict[str, float]:
    """Full dual-domain metric bundle for one prediction/target pair."""
    out: Dict[str, float] = {}
    out["mse"] = mse_defect_maps(pred_svg, target_svg, size)
    out.update(iou_precision_recall_f1(pred_svg, target_svg, size))

    pred_points = sample_svg_defect_points(pred_svg)
    target_points = sample_svg_defect_points(target_svg)
    out["dtw"] = dtw_average_step_cost(pred_points, target_points)
    out["ldtw"] = dtw_average_step_cost(pred_points, target_points, window_ratio=ldtw_window_ratio)
    out["cd"] = chamfer_distance(pred_points, target_points)
    return out


class MetricAverager:
    """Running mean of metric dicts (NaNs are skipped per metric)."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            self._sums[k] = self._sums.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0) + 1

    def compute(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}
