"""Chỉ số đánh giá: Recall@K, MRR, CI bootstrap, McNemar, AUROC, ECE, Brier."""
from __future__ import annotations

from math import comb

import numpy as np


def first_gold_rank(doc_ids: list[str], gold: set[str]) -> int | None:
    return next((i for i, d in enumerate(doc_ids, 1) if d in gold), None)


def retrieval_summary(ranks: list[int | None], ks=(1, 3, 5, 10)) -> dict:
    n = max(len(ranks), 1)
    out = {f"R@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out["MRR@10"] = sum(1 / r for r in ranks if r and r <= 10) / n
    return out


def hits_at(ranks: list[int | None], k: int) -> list[bool]:
    return [r is not None and r <= k for r in ranks]


def bootstrap_ci(values, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def mcnemar_p(a_hits, b_hits) -> tuple[float, int, int]:
    """Kiểm định McNemar chính xác. Trả về (p, số câu a đúng b sai, số câu b đúng a sai)."""
    b = sum(bool(x) and not y for x, y in zip(a_hits, b_hits))
    c = sum(bool(y) and not x for x, y in zip(a_hits, b_hits))
    n = b + c
    if n == 0:
        return 1.0, b, c
    p = 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, p), b, c


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def brier(probs, labels) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((p - y) ** 2))


def ece(probs, labels, n_bins: int = 10) -> float:
    """Expected Calibration Error: trung bình |độ tin cậy dự đoán − tỷ lệ đúng thực tế| theo các ngăn."""
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)
