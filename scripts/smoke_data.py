"""Smoke test trên dữ liệu thật (chạy trong GitHub Actions, không cần GPU / model nơ-ron).

Kiểm tra: dựng kho tài liệu giống BTC, độ chính xác khôi phục dấu trên 100 câu dev,
Recall BM25 trên câu dev có dấu / không dấu / không dấu đã khôi phục / teencode đã chuẩn hóa.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vicare_safe.data import dev_set, load_kb  # noqa: E402
from vicare_safe.diacritics import syllable_accuracy  # noqa: E402
from vicare_safe.metrics import first_gold_rank, retrieval_summary  # noqa: E402
from vicare_safe.pipeline import build_restorer  # noqa: E402
from vicare_safe.retrieval import BM25  # noqa: E402
from vicare_safe.text import fold, normalize_teencode, to_teencode  # noqa: E402

kb = load_kb()
dev = dev_set(kb)
assert len(kb.corpus) == 9895 and len(kb.chunks) == 11227 and len(dev) == 100, "Kho / dev khác notebook BTC"

restorer = build_restorer(kb)
acc = syllable_accuracy(restorer, [q["question"] for q in dev])

bm = BM25([c["text"] for c in kb.chunks])
doc_of = [c["doc_id"] for c in kb.chunks]


def bm25_recall(queries):
    ranks = []
    for q, item in zip(queries, dev):
        order = bm.scores(q).argsort()[::-1]
        docs, seen = [], set()
        for i in order[:200]:
            if doc_of[i] not in seen:
                seen.add(doc_of[i]); docs.append(doc_of[i])
        ranks.append(first_gold_rank(docs[:10], set(item["gold_doc_ids"])))
    return retrieval_summary(ranks)


report = {
    "restoration": acc,
    "bm25_dev": bm25_recall([q["question"] for q in dev]),
    "bm25_dev_nodau": bm25_recall([fold(q["question"]) for q in dev]),
    "bm25_dev_nodau_restored": bm25_recall([restorer.restore(fold(q["question"])) for q in dev]),
    "bm25_dev_teencode_normalized": bm25_recall([normalize_teencode(to_teencode(q["question"]))[0] for q in dev]),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
assert acc["syllable_acc"] >= 0.90, "Khôi phục dấu kém hơn ngưỡng 90%"
assert report["bm25_dev"]["R@10"] >= 0.40, "BM25 Recall@10 thấp bất thường"
print("SMOKE OK")
