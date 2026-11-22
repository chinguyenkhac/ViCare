import numpy as np

from vicare_safe.retrieval import BM25, HybridRetriever, bm25_tokens, embed_passages


def test_bm25_tokens_bigrams():
    assert bm25_tokens("Viêm gan B") == ["viem", "gan", "b", "viem_gan", "gan_b"]


def test_bm25_ranking_accent_insensitive(chunks):
    bm = BM25([c["text"] for c in chunks])
    s = bm.scores("viem gan b lay qua duong nao")
    assert int(np.argmax(s)) == 0


def test_hybrid_and_dedupe(chunks, embedder):
    emb = embed_passages(embedder, [c["text"] for c in chunks])
    r = HybridRetriever(chunks, emb, embedder, reranker=None, n_each=4)
    hits = r.hybrid("tre sot cao co giat", k=3)
    assert hits[0]["doc_id"] == "d2"
    assert len({h["doc_id"] for h in hits}) == len(hits)
    assert {"dense_rank", "bm25_rank", "dense_cos", "rrf"} <= set(hits[0])
    assert r.dense("Viêm gan B lây qua đường máu", k=2)[0]["doc_id"] in {"d1", "d3"}


class FakeReranker:
    def score(self, query, texts):
        return np.array([0.9 if "ruột thừa" in t else 0.1 for t in texts], dtype=np.float32)


def test_rerank_uses_reranker(chunks, embedder):
    emb = embed_passages(embedder, [c["text"] for c in chunks])
    r = HybridRetriever(chunks, emb, embedder, reranker=FakeReranker(), n_each=4)
    hits = r.rerank("dau bung", k=2, n_cand=4)
    assert hits[0]["doc_id"] == "d4" and hits[0]["score"] == np.float32(0.9)
