"""Tính sẵn embedding e5-small cho toàn bộ chunk của kho tài liệu (lưu float16).

Chạy:  python scripts/build_embeddings.py
Kết quả: artifacts/e5_small_chunks.f16.npy + artifacts/chunk_ids.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vicare_safe.data import load_kb  # noqa: E402
from vicare_safe.retrieval import EMBEDDING_MODEL, embed_passages, load_embedder  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts"

if __name__ == "__main__":
    kb = load_kb()
    texts = [c["text"] for c in kb.chunks]
    print(f"{len(kb.corpus):,} tài liệu → {len(texts):,} chunk", flush=True)
    model = load_embedder()
    t0 = time.time()
    vecs = embed_passages(model, texts, show_progress=True)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "e5_small_chunks.f16.npy", vecs.astype(np.float16))
    (OUT / "chunk_ids.json").write_text(json.dumps({"model": EMBEDDING_MODEL, "ids": [c["id"] for c in kb.chunks]}))
    print(f"Xong {len(texts):,} chunk trong {time.time() - t0:.0f}s", flush=True)
