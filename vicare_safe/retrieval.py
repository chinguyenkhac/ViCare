"""Truy xuất lai: vector (e5) + BM25 (bỏ dấu, bigram âm tiết) → RRF → reranker cross-encoder.

Không dùng vector DB: 11k chunk × 384 chiều vừa một ma trận numpy (~17 MB), nhân ma trận nhanh hơn
và dễ đóng gói lên Hugging Face Space. BM25 dùng ma trận thưa (scipy) thay cho rank-bm25 (nhanh ~100×,
cùng công thức Okapi).
"""
from __future__ import annotations

import hashlib
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .text import fold

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


# ----------------------------------------------------------------------------- vector
def load_embedder(device: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device=device or _device())


def embed_passages(model, texts, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    return model.encode(["passage: " + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=show_progress, convert_to_numpy=True).astype(np.float32)


def embed_queries(model, questions) -> np.ndarray:
    return model.encode(["query: " + q for q in questions], normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


# ----------------------------------------------------------------------------- BM25
def bm25_tokens(text: str) -> list[str]:
    """Âm tiết bỏ dấu + bigram âm tiết ('viem_gan') – giữ từ ghép mà không cần tách từ."""
    w = re.findall(r"\w+", fold(text))
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


class BM25:
    """BM25 Okapi (k1=1.5, b=0.75, sàn idf = 0.25 × idf trung bình – như rank-bm25) trên ma trận thưa."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        docs = [Counter(bm25_tokens(t)) for t in texts]
        self.vocab: dict[str, int] = {}
        df = Counter()
        for d in docs:
            df.update(d.keys())
        for tok in df:
            self.vocab[tok] = len(self.vocab)
        n = len(docs)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for tok, i in self.vocab.items():
            idf[i] = math.log(n - df[tok] + 0.5) - math.log(df[tok] + 0.5)
        idf[idf < 0] = epsilon * float(idf.mean())
        dl = np.array([sum(d.values()) for d in docs], dtype=np.float32)
        norm = k1 * (1 - b + b * dl / dl.mean())
        rows, cols, vals = [], [], []
        for r, d in enumerate(docs):
            for tok, tf in d.items():
                c = self.vocab[tok]
                rows.append(r)
                cols.append(c)
                vals.append(idf[c] * tf * (k1 + 1) / (tf + norm[r]))
        self.matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)), dtype=np.float32)

    def scores(self, query: str) -> np.ndarray:
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for tok, cnt in Counter(bm25_tokens(query)).items():
            i = self.vocab.get(tok)
            if i is not None:
                q[i] = cnt
        return np.asarray(self.matrix @ q).ravel()


def _topn(scores: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(scores))
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx], kind="stable")]


# ----------------------------------------------------------------------------- reranker
class Reranker:
    """Cross-encoder có cache điểm theo (câu hỏi, đoạn văn) – chạy lại notebook không phải tính lại."""

    def __init__(self, model_name: str = RERANK_MODEL, max_length: int = 320, device: str | None = None,
                 cache_path: str | Path | None = None, batch_size: int = 16):
        from sentence_transformers import CrossEncoder
        dev = device or _device()
        kwargs = {}
        if dev == "cuda":
            import torch
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = CrossEncoder(model_name, max_length=max_length, device=dev, **kwargs)
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[tuple[str, str], float] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = pickle.loads(self.cache_path.read_bytes())

    @staticmethod
    def _key(query: str, text: str) -> tuple[str, str]:
        return query, hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        keys = [self._key(query, t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            pred = self.model.predict([(query, texts[i]) for i in todo], batch_size=self.batch_size,
                                      show_progress_bar=False)
            for i, s in zip(todo, np.atleast_1d(pred)):
                self.cache[keys[i]] = float(s)
        return np.array([self.cache[k] for k in keys], dtype=np.float32)

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps(self.cache))


# ----------------------------------------------------------------------------- hybrid
class HybridRetriever:
    """dense / bm25 / hybrid (RRF) / hybrid + rerank. Mỗi kết quả là 1 tài liệu (chunk tốt nhất)."""

    def __init__(self, chunks: list[dict], embeddings: np.ndarray, embedder, reranker: Reranker | None = None,
                 rrf_k: int = 60, n_each: int = 60):
        self.chunks = chunks
        self.emb = embeddings.astype(np.float32)
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.n_each = n_each
        self.bm25 = BM25([c["text"] for c in chunks])
        self._qcache: dict[str, np.ndarray] = {}

    # -- thành phần
    def qvec(self, query: str) -> np.ndarray:
        if query not in self._qcache:
            self._qcache[query] = embed_queries(self.embedder, [query])[0]
        return self._qcache[query]

    def _dense_ranked(self, query: str, n: int):
        s = self.emb @ self.qvec(query)
        idx = _topn(s, n)
        return idx, s

    def _bm25_ranked(self, query: str, n: int):
        s = self.bm25.scores(query)
        return _topn(s, n), s

    def _hit(self, i: int, score: float, **extra) -> dict:
        c = self.chunks[i]
        return {"chunk_idx": int(i), "chunk_id": c["id"], "doc_id": c["doc_id"], "url": c["url"],
                "text": c["text"], "score": float(score), **extra}

    @staticmethod
    def _dedupe(ranked: list[dict], k: int) -> list[dict]:
        out, seen = [], set()
        for h in ranked:
            if h["doc_id"] in seen:
                continue
            seen.add(h["doc_id"])
            out.append(h)
            if len(out) >= k:
                break
        return out

    # -- các bộ truy xuất
    def dense(self, query: str, k: int = 5) -> list[dict]:
        """Baseline BTC: chỉ vector, score = cosine."""
        idx, s = self._dense_ranked(query, max(k * 10, 30))
        return self._dedupe([self._hit(i, s[i]) for i in idx], k)

    def bm25_only(self, query: str, k: int = 5) -> list[dict]:
        idx, s = self._bm25_ranked(query, max(k * 10, 50))
        return self._dedupe([self._hit(i, s[i]) for i in idx], k)

    def hybrid(self, query: str, k: int = 5) -> list[dict]:
        n = max(k * 3, self.n_each)
        d_idx, d_s = self._dense_ranked(query, n)
        b_idx, _ = self._bm25_ranked(query, n)
        fused: dict[int, float] = {}
        for ranking in (d_idx, b_idx):
            for r, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.rrf_k + r + 1)
        # hạng theo tài liệu (để làm đặc trưng độ tin cậy)
        d_rank = self._doc_ranks(d_idx)
        b_rank = self._doc_ranks(b_idx)
        ranked = sorted(fused, key=fused.get, reverse=True)
        hits = []
        for i in ranked:
            doc = self.chunks[i]["doc_id"]
            hits.append(self._hit(i, fused[i], rrf=fused[i], dense_cos=float(d_s[i]),
                                  dense_rank=d_rank.get(doc), bm25_rank=b_rank.get(doc)))
        return self._dedupe(hits, k)

    def _doc_ranks(self, idx) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for i in idx:
            doc = self.chunks[int(i)]["doc_id"]
            if doc not in ranks:
                ranks[doc] = len(ranks) + 1
        return ranks

    def rerank(self, query: str, k: int = 5, n_cand: int = 10) -> list[dict]:
        """Hybrid → n_cand ứng viên → cross-encoder. score ∈ [0,1] = mức liên quan."""
        cands = self.hybrid(query, k=max(n_cand, k))
        if not cands or self.reranker is None:
            return cands[:k]
        scores = self.reranker.score(query, [c["text"] for c in cands])
        for c, s in zip(cands, scores):
            c["score"] = float(s)
        return sorted(cands, key=lambda c: -c["score"])[:k]
