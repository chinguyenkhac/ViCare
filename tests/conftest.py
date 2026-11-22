import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vicare_safe.text import fold  # noqa: E402


class HashEmbedder:
    """Embedder giả lập cho test: túi âm tiết bỏ dấu → vector chuẩn hóa (không cần tải model)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, **_):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in fold(t.split(": ", 1)[-1]).split():
                out[r, hash(w) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


CHUNKS = [
    {"id": "d1#0", "doc_id": "d1", "url": "https://x/1", "text": "Viêm gan B lây truyền qua đường máu, quan hệ tình dục và từ mẹ sang con."},
    {"id": "d2#0", "doc_id": "d2", "url": "https://x/2", "text": "Trẻ sốt cao co giật cần được đưa đến cơ sở y tế ngay. Chườm ấm và theo dõi."},
    {"id": "d3#0", "doc_id": "d3", "url": "https://x/3", "text": "Người bị viêm gan B mạn tính ổn định vẫn có thể tiêm vaccine Covid-19."},
    {"id": "d4#0", "doc_id": "d4", "url": "https://x/4", "text": "Đau bụng bên phải dưới có thể là dấu hiệu viêm ruột thừa, cần đi khám."},
]


@pytest.fixture
def chunks():
    return [dict(c) for c in CHUNKS]


@pytest.fixture
def embedder():
    return HashEmbedder()
