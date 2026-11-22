import numpy as np

from vicare_safe.confidence import ConfidenceModel
from vicare_safe.diacritics import DiacriticRestorer
from vicare_safe.pipeline import ViCareSafe, is_followup
from vicare_safe.retrieval import HybridRetriever, embed_passages


class FakeReranker:
    def score(self, query, texts):
        q = set(query.lower().split())
        return np.array([min(1.0, len(q & set(t.lower().split())) / 4) for t in texts], dtype=np.float32)


def _app(chunks, embedder, tau=0.3):
    emb = embed_passages(embedder, [c["text"] for c in chunks])
    retr = HybridRetriever(chunks, emb, embedder, reranker=FakeReranker(), n_each=4)
    restorer = DiacriticRestorer().fit([c["text"] for c in chunks] + ["con tôi bị sốt cao co giật"] * 3)
    conf = ConfidenceModel()
    conf.mu, conf.sd, conf.w = np.zeros(8), np.ones(8), np.r_[0.0, 3.0, np.zeros(7)]
    return ViCareSafe(retr, restorer, conf, tau=tau, n_cand=4)


def test_unaccented_emergency_question(chunks, embedder):
    a = _app(chunks, embedder).ask("tre sot cao co giat phai lam sao")
    assert a.emergency and a.understood.restored
    assert "115" in a.markdown and "khôi phục dấu" in a.markdown


def test_oos_refused(chunks, embedder):
    a = _app(chunks, embedder, tau=0.99).ask("Giá vàng hôm nay bao nhiêu?")
    assert a.status == "refused_oos" and "không tìm thấy" in a.markdown


def test_dose_blocked(chunks, embedder):
    a = _app(chunks, embedder).ask("Trẻ 15kg uống paracetamol liều bao nhiêu?")
    assert a.status == "dose_blocked" and "không đưa liều" in a.markdown


def test_answer_has_confidence_evidence_sources(chunks, embedder):
    a = _app(chunks, embedder).ask("Viêm gan B lây truyền qua đường máu không?")
    assert a.status in ("answered", "low_confidence")
    assert "Độ tin cậy" in a.markdown and "Nguồn" in a.markdown and a.evidence


def test_followup():
    assert is_followup("Vậy trường hợp này có nguy hiểm không?")
    assert not is_followup("Con tôi bị sốt thế nào thì cần đi viện?")
