"""Trích đoạn bằng chứng: chọn câu nguyên văn trong tài liệu bác sĩ làm căn cứ cho câu trả lời."""
from __future__ import annotations

from .text import is_boilerplate, split_sentences


def candidate_sentences(text: str, max_sents: int = 8) -> list[str]:
    sents = [s for s in split_sentences(text) if not is_boilerplate(s)]
    return sents[:max_sents]


def extract_evidence(query: str, hits: list[dict], reranker, n_docs: int = 3, max_sents: int = 8,
                     min_score: float = 0.05) -> list[dict]:
    """Mỗi nguồn (top n_docs) lấy 1 câu liên quan nhất; trả về theo thứ tự nguồn [1], [2], [3]."""
    pool = []
    for n, h in enumerate(hits[:n_docs], 1):
        for s in candidate_sentences(h["text"], max_sents):
            pool.append((n, s))
    if not pool:
        return []
    if reranker is None:
        scores = [1.0 / (i + 1) for i in range(len(pool))]
    else:
        scores = reranker.score(query, [s for _, s in pool]).tolist()
    best: dict[int, dict] = {}
    for (n, s), sc in zip(pool, scores):
        if n not in best or sc > best[n]["score"]:
            best[n] = {"source": n, "sentence": s, "score": float(sc)}
    return [best[n] for n in sorted(best) if best[n]["score"] >= min_score]
