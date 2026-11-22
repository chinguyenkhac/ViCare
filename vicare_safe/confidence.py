"""Độ tin cậy của câu trả lời: hồi quy logistic trên các tín hiệu truy xuất.

Định nghĩa (quan trọng khi trình bày): độ tin cậy = xác suất ước lượng rằng **tài liệu gốc trả lời đúng câu hỏi
nằm trong 3 nguồn ViCare hiển thị**. Đây là độ tin về việc *tìm đúng tài liệu*, KHÔNG phải độ đúng y khoa.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .text import syllables

FEATURES = ["rerank_top1", "rerank_margin", "rerank_top3_mean", "agree_bm25_dense", "dense_cos_top1",
            "restored", "normalized", "log_len"]

LEVEL_HIGH, LEVEL_MID = 0.70, 0.40
LEVELS = ("Cao", "Trung bình", "Thấp")


def features(hits: list[dict], query: str, restored: bool = False, normalized: bool = False) -> list[float]:
    s = [h["score"] for h in hits] + [0.0, 0.0, 0.0]
    top = hits[0] if hits else {}
    d_rank, b_rank = top.get("dense_rank") or 99, top.get("bm25_rank") or 99
    return [
        s[0],
        s[0] - s[1],
        float(np.mean(s[:3])),
        1.0 if (d_rank <= 5 and b_rank <= 5) else 0.0,
        float(top.get("dense_cos", 0.0)),
        float(restored),
        float(normalized),
        math.log1p(len(syllables(query))),
    ]


def level_of(p: float) -> str:
    return LEVELS[0] if p >= LEVEL_HIGH else LEVELS[1] if p >= LEVEL_MID else LEVELS[2]


class ConfidenceModel:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def _x(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        Z = (X - self.mu) / self.sd
        return np.hstack([np.ones((len(Z), 1)), Z])

    def fit(self, X, y, iters: int = 50) -> "ConfidenceModel":
        """Hồi quy logistic có L2, tối ưu bằng Newton (IRLS) – vài trăm mẫu, hội tụ tức thì."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        A = self._x(X)
        w = np.zeros(A.shape[1])
        reg = np.eye(len(w)) * self.l2
        reg[0, 0] = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - y) + reg @ w
            H = A.T @ (A * (p * (1 - p))[:, None]) + reg
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict_proba(self, X) -> np.ndarray:
        return 1 / (1 + np.exp(-self._x(X) @ self.w))

    def predict_one(self, feats: list[float]) -> float:
        return float(self.predict_proba([feats])[0])

    def coefficients(self) -> dict[str, float]:
        return {"bias": float(self.w[0]), **{f: float(c) for f, c in zip(FEATURES, self.w[1:])}}

    # -------------------------------------------------------------- lưu / nạp (cho Space)
    def to_dict(self) -> dict:
        return {"features": FEATURES, "w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "levels": {"high": LEVEL_HIGH, "mid": LEVEL_MID}}

    def save(self, path: str | Path, **extra) -> None:
        Path(path).write_text(json.dumps({**self.to_dict(), **extra}, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> tuple["ConfidenceModel", dict]:
        d = json.loads(Path(path).read_text())
        m = cls()
        m.w, m.mu, m.sd = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"])
        return m, d


def pick_oos_threshold(in_scores, oos_scores, target: float = 0.95) -> float:
    """Ngưỡng nhỏ nhất sao cho ≥ target câu ngoài phạm vi bị từ chối (thận trọng – như v2)."""
    for t in sorted(set(in_scores) | set(oos_scores) | {1.01}):
        if np.mean([s < t for s in oos_scores]) >= target:
            return float(t)
    return 1.01


def reliability_by_level(probs, labels) -> list[dict]:
    rows = []
    for name in LEVELS:
        idx = [i for i, p in enumerate(probs) if level_of(p) == name]
        rows.append({"level": name, "n": len(idx),
                     "mean_conf": float(np.mean([probs[i] for i in idx])) if idx else float("nan"),
                     "actual_acc": float(np.mean([labels[i] for i in idx])) if idx else float("nan")})
    return rows


def norm_reliability(rows: list[dict]) -> list[dict]:
    """Chuẩn hóa khóa bảng độ tin cậy (notebook lưu tên cột tiếng Việt)."""
    alias = {"số câu": "n", "độ tin cậy dự đoán TB": "mean_conf", "tỷ lệ tìm đúng thực tế": "actual_acc"}
    return [{alias.get(k, k): v for k, v in r.items()} for r in rows]
