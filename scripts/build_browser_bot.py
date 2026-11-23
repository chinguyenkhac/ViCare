"""Cấu hình + đánh giá cho chatbot chạy trong trình duyệt (GitHub Pages, bản nhẹ chỉ dùng BM25).

Trình duyệt tự tải ViHealthQA từ Hugging Face (repo không phát hành lại dữ liệu), dựng đúng BM25 như
vicare_safe.retrieval.BM25, rồi dùng ngưỡng từ chối + mô hình độ tin cậy được chọn ở đây (trên tập train / OOS_TUNE,
không dùng câu dev). Kết quả trên 100 câu dev của BTC được ghi vào docs/bot/config.json để trang hiển thị trung thực.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vicare_safe import pipeline as P  # noqa: E402
from vicare_safe import safety as S  # noqa: E402
from vicare_safe import text as T  # noqa: E402
from vicare_safe.data import OOS_TEST, OOS_TUNE, dev_set, load_kb, tune_set  # noqa: E402
from vicare_safe.retrieval import BM25, bm25_tokens  # noqa: E402

LEVELS = (0.70, 0.40)
OOS_TARGET = float(__import__("os").environ.get("OOS_TARGET", "0.80"))


def doc_hits(bm25, kb, query, k=5, pool=50):
    s = bm25.scores(query)
    idx = np.argsort(-s, kind="stable")[:pool]
    hits, seen = [], set()
    for i in idx:
        d = kb.chunks[i]["doc_id"]
        if d in seen:
            continue
        seen.add(d)
        hits.append((d, float(s[i])))
        if len(hits) == k:
            break
    return hits


def feats(hits, query):
    s1 = hits[0][1] if hits else 0.0
    s2 = hits[1][1] if len(hits) > 1 else 0.0
    n = max(1, len(bm25_tokens(query)))
    return [math.log1p(s1), (s1 - s2) / s1 if s1 > 0 else 0.0, s1 / n]


def gate_feats(bm25, hits, query):
    """Đặc trưng 'có thuộc lĩnh vực sức khỏe không': điểm top-1, tỷ lệ cặp âm tiết có trong kho, tỷ lệ từ lạ."""
    import re
    w = re.findall(r"\w+", T.fold(query))
    big = [a + "_" + b for a, b in zip(w, w[1:])]
    bigfrac = sum(t in bm25.vocab for t in big) / max(1, len(big))
    oov = sum(t not in bm25.vocab for t in w) / max(1, len(w))
    s1 = hits[0][1] if hits else 0.0
    return [math.log1p(s1), bigfrac, oov]


def prep(q):
    t, _ = T.normalize_teencode(q)
    return t


def fit_logistic(X, y, l2=1.0, iters=50):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0])
        w -= np.linalg.solve(H, g)
    return mu, sd, w


def proba(model, x):
    mu, sd, w = model
    z = np.r_[(np.asarray(x) - mu) / sd, 1.0] @ w
    return float(1 / (1 + math.exp(-z)))


def main():
    kb = load_kb()
    print("splits:", list(kb.rows), "| docs", len(kb.corpus), "| chunks", len(kb.chunks))
    bm25 = BM25([c["text"] for c in kb.chunks])

    # --- huấn luyện trên train (200 câu) + OOS_TUNE; nhãn = tài liệu gốc nằm trong 3 nguồn hiển thị
    tune = tune_set(kb, 200)
    X, y, G, gy = [], [], [], []
    for q in tune:
        for variant in (q["question"], T.fold(q["question"])):
            qq = prep(variant)
            h = doc_hits(bm25, kb, qq)
            X.append(feats(h, qq))
            y.append(float(any(d in q["gold_doc_ids"] for d, _ in h[:3])))
            G.append(gate_feats(bm25, h, qq))
            gy.append(1.0)
    for q in OOS_TUNE:
        for variant in (q, T.fold(q)):
            qq = prep(variant)
            h = doc_hits(bm25, kb, qq)
            G.append(gate_feats(bm25, h, qq))
            gy.append(0.0)
    X, y = np.array(X), np.array(y)
    model = fit_logistic(X, y)
    gate = fit_logistic(np.array(G), np.array(gy))
    oos_p = sorted(proba(gate, g) for g, t in zip(G, gy) if t == 0)
    tau = float(oos_p[int(math.ceil(OOS_TARGET * len(oos_p))) - 1]) + 1e-6  # từ chối ≥ OOS_TARGET câu OOS_TUNE

    def decide(q, history=None):
        if history and P.is_followup(q):
            q = history[-1] + " " + q
        qq = prep(q)
        h = doc_hits(bm25, kb, qq)
        f = feats(h, qq)
        if proba(gate, gate_feats(bm25, h, qq)) < tau:
            return "Từ chối", h, None
        p = proba(model, f)
        lvl = "Cao" if p >= LEVELS[0] else "Trung bình" if p >= LEVELS[1] else "Thấp"
        return lvl, h, p

    # --- đánh giá trên 100 câu dev BTC (không dùng khi chọn ngưỡng)
    dev = dev_set(kb)
    variants = {
        "có dấu": [q["question"] for q in dev],
        "không dấu": [T.fold(q["question"]) for q in dev],
        "teencode + không dấu": [T.fold(T.to_teencode(q["question"])) for q in dev],
    }
    results, rel = {}, {"Cao": [], "Trung bình": [], "Thấp": []}
    for name, qs in variants.items():
        r1 = r5 = 0
        dec = Counter()
        for q, item in zip(qs, dev):
            lvl, h, p = decide(q)
            ids = [d for d, _ in h]
            r1 += ids[:1] and ids[0] in item["gold_doc_ids"]
            r5 += any(d in item["gold_doc_ids"] for d in ids[:5])
            dec[lvl] += 1
            if lvl in rel:
                rel[lvl].append(float(any(d in item["gold_doc_ids"] for d in ids[:3])))
        results[name] = {"R@1": r1 / len(dev), "R@5": r5 / len(dev),
                         "decisions": {k: dec.get(k, 0) / len(dev) for k in ("Cao", "Trung bình", "Thấp", "Từ chối")}}
        print(f"{name:22s} R@1={r1}% R@5={r5}% | {dict(dec)}")
    oos_ref = [decide(q)[0] == "Từ chối" for q in OOS_TEST] + [decide(T.fold(q))[0] == "Từ chối" for q in OOS_TEST]
    reliability = {k: {"n": len(v), "actual": (sum(v) / len(v)) if v else None} for k, v in rel.items()}
    print("OOS_TEST refused:", sum(oos_ref), "/", len(oos_ref), "| reliability", reliability)
    print("tau", tau, "| weights", model[2])

    cfg = {
        "dataset": "tarudesu/ViHealthQA",
        "parquet_api": "https://huggingface.co/api/datasets/tarudesu/ViHealthQA/parquet",
        "splits": list(kb.rows),
        "chunk": {"size": 200, "overlap": 40},
        "bm25": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
        "teencode": T.TEENCODE, "letter_guard": sorted(T._LETTER_GUARD),
        "emerg_accent": S.EMERG_ACCENT, "emerg_folded": S.EMERG_FOLDED,
        "dose_accent": S.DOSE_ACCENT, "dose_folded": S.DOSE_FOLDED,
        "followup_start": list(P.FOLLOWUP_START), "followup_any": P.FOLLOWUP_ANY,
        "emergency_msg": S.EMERGENCY_MSG, "dose_msg": S.DOSE_MSG,
        "model": {"mu": model[0].tolist(), "sd": model[1].tolist(), "w": model[2].tolist()},
        "gate": {"mu": gate[0].tolist(), "sd": gate[1].tolist(), "w": gate[2].tolist()},
        "tau": tau, "levels": list(LEVELS),
        "eval": {"dev": results, "oos_test_refused": sum(oos_ref) / len(oos_ref), "reliability": reliability,
                 "n_dev": len(dev), "n_tune": len(tune)},
        "corpus": {"docs": len(kb.corpus), "chunks": len(kb.chunks)},
    }
    out = ROOT / "docs/bot/config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    # mẫu kiểm tra chéo Python ↔ JS: top-5 tài liệu + điểm cho vài câu
    probes = ["con toi 2 tuoi sot cao co giat phai lam sao", "Đau bụng bên phải dưới có phải viêm ruột thừa không",
              "bs ơi bà bầu bị cúm uống thuốc hạ sốt dc ko", "Giá vàng hôm nay bao nhiêu?"]
    check = []
    for q in probes:
        lvl, h, p = decide(q)
        check.append({"q": q, "level": lvl, "p": p, "gate_p": proba(gate, gate_feats(bm25, h, prep(q))),
                      "top": [[d, round(s, 4)] for d, s in h]})
    (ROOT / "docs/bot/check.json").write_text(json.dumps(check, ensure_ascii=False, indent=1))
    print("✅ docs/bot/config.json + check.json")


if __name__ == "__main__":
    main()
