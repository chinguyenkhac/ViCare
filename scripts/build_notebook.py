"""Sinh notebook đánh giá notebooks/ViCare_Safe_v3.ipynb từ code trong repo (một nguồn sự thật duy nhất)."""
from pathlib import Path

import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# ViCare-Safe v3 – Hiểu tiếng Việt không dấu, có độ tin cậy và bằng chứng

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chinguyenkhac/ViCare/blob/main/notebooks/ViCare_Safe_v3.ipynb)

**Bài dự thi AI Challenge 24h** – phát triển từ notebook ViCare của BTC (v1) và ViCare-Safe v2.
Notebook này **đánh giá trước/sau** các cải tiến P0 trên cùng dữ liệu, cùng 100 câu dev của BTC:

| Cải tiến | Vấn đề thực tế | Đo bằng |
|---|---|---|
| **Khôi phục dấu** (trigram âm tiết + Viterbi, không LLM) | v2 tìm được tài liệu cho câu không dấu nhưng **từ chối nhầm 80%** vì reranker chấm thấp chữ không dấu | % âm tiết đúng, Recall@k, tỷ lệ bị từ chối |
| **Chuẩn hóa teencode** | "ko", "j", "dc", "bs"… làm lệch truy xuất | Recall@k trên dev teencode |
| **Độ tin cậy** (hồi quy logistic) | Một ngưỡng duy nhất chỉ có "trả lời" hoặc "từ chối" | Độ chính xác thực tế theo mức, ECE, AUROC |
| **Trích đoạn bằng chứng** | Người dùng khó tự kiểm chứng | Ví dụ đầu ra |

**Độ tin cậy được định nghĩa là:** xác suất *tài liệu gốc trả lời câu hỏi nằm trong 3 nguồn ViCare hiển thị* – độ tin về **tìm đúng tài liệu**, không phải độ đúng y khoa.

> ⚠️ Demo học thuật, **không** chẩn đoán, **không** kê đơn. Dữ liệu ViHealthQA chỉ dùng cho mục đích nghiên cứu.

**Chạy:** Colab → `Runtime → Run all` (GPU T4 nhanh hơn; CPU vẫn chạy được nhờ embedding tính sẵn trong repo).
""")

md("## 0. Cài đặt")
code(r"""
import os, sys, subprocess, time, json
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not Path("vicare_safe").exists():
    if not Path("ViCare").exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/chinguyenkhac/ViCare"], check=True)
    os.chdir("ViCare")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
ROOT = Path.cwd() if (Path.cwd() / "vicare_safe").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
print("Thư mục dự án:", ROOT)
""")
code(r"""
# ===== Cấu hình (khớp với demo trên Hugging Face Space – CPU 2 nhân) =====
N_CAND = 10            # số ứng viên đưa vào reranker
RERANK_MAX_LEN = 320
OOS_TARGET = 0.95      # cổng: từ chối ≥95% câu ngoài phạm vi trên tập tune (thận trọng, như v2)
N_TUNE = 200          # câu train để chọn ngưỡng và huấn luyện độ tin cậy
""")

md("## 1. Dữ liệu – kho tài liệu và tập dev giống hệt BTC")
code(r"""
import numpy as np, pandas as pd
from vicare_safe.data import load_kb, dev_set, tune_set, OOS_TUNE, OOS_TEST
from vicare_safe.text import fold, to_teencode, normalize_teencode

kb = load_kb()
dev, tune = dev_set(kb), tune_set(kb, N_TUNE)
print(f"Kho: {len(kb.corpus):,} tài liệu → {len(kb.chunks):,} chunk | dev: {len(dev)} | tune: {len(tune)}")
print(f"Câu ngoài phạm vi: tune {len(OOS_TUNE)} | test {len(OOS_TEST)}")
""")

md(r"""
## 2. P0.1 – Khôi phục dấu tiếng Việt

Học tần suất âm tiết có dấu (unigram + bigram + trigram) từ kho tài liệu và câu hỏi tập train (văn phong người hỏi, nhân trọng số 3),
rồi giải mã Viterbi bậc 2. (Trên 400 câu validation *không thuộc* dev: trigram 97,9% âm tiết / 74,8% câu đúng; bigram 97,6% / 71,3%.) Âm tiết người dùng đã gõ dấu được giữ nguyên. **Không dùng câu hỏi dev/validation để học.**
""")
code(r"""
from vicare_safe.pipeline import build_restorer
from vicare_safe.diacritics import syllable_accuracy

t0 = time.time()
restorer = build_restorer(kb)
print(f"Học xong trong {time.time()-t0:.1f}s | {len(restorer.uni):,} âm tiết | {len(restorer.bi):,} bigram | {len(restorer.tri):,} trigram")
restoration = syllable_accuracy(restorer, [q["question"] for q in dev])
print(f"Trên 100 câu dev: {restoration['syllable_acc']:.1%} âm tiết đúng | {restoration['sentence_exact']:.0%} câu đúng hoàn toàn")
for q in dev[:6]:
    print(f"  {fold(q['question'])[:70]:72} → {restorer.restore(fold(q['question']))[:70]}")
""")

md(r"""
## 3. P0.2 – Chuẩn hóa teencode

Dữ liệu gốc gần như không có teencode (0,6% câu hỏi), nên dùng **bản dev teencode hóa tổng hợp** (ví dụ *không → ko, được → dc, gì → j, bác sĩ → bs*).
""")
code(r"""
dev_teen = [{**q, "question": to_teencode(q["question"])} for q in dev]
changed = sum(t["question"] != q["question"] for t, q in zip(dev_teen, dev))
print(f"{changed}/100 câu dev có ít nhất một từ bị teencode hóa")
for q in dev_teen[:4]:
    print(f"  {q['question'][:70]:72} → {normalize_teencode(q['question'])[0][:70]}")
""")

md(r"""
## 4. Chỉ mục truy xuất: vector e5 (tính sẵn) + BM25 + reranker

Embedding 11.227 chunk được tính sẵn bằng `scripts/build_embeddings.py` và lưu trong `artifacts/` (float16).
Nếu chưa có, notebook tự tính lại (GPU ~1 phút, CPU ~10–15 phút).
""")
code(r"""
from vicare_safe.retrieval import HybridRetriever, Reranker, load_embedder, embed_passages

embedder = load_embedder()
emb_path, ids_path = ROOT / "artifacts/e5_small_chunks.f16.npy", ROOT / "artifacts/chunk_ids.json"
ids = [c["id"] for c in kb.chunks]
if emb_path.exists() and json.loads(ids_path.read_text())["ids"] == ids:
    emb = np.load(emb_path).astype(np.float32)
    print("Nạp embedding tính sẵn:", emb.shape)
else:
    emb = embed_passages(embedder, [c["text"] for c in kb.chunks], show_progress=True)
reranker = Reranker(max_length=RERANK_MAX_LEN, cache_path=ROOT / "cache/rerank_scores.pkl")
retr = HybridRetriever(kb.chunks, emb, embedder, reranker)
print("BM25 vocab:", len(retr.bm25.vocab))
""")

md(r"""
## 5. Chạy truy xuất trên mọi biến thể câu hỏi

| Biến thể | Mô tả |
|---|---|
| `có dấu` | câu dev gốc |
| `không dấu – thô` | bỏ dấu, đưa thẳng vào hệ thống (như v2) |
| `không dấu – khôi phục` | bỏ dấu → **khôi phục dấu** (P0.1) |
| `teencode – thô` | teencode hóa, đưa thẳng vào hệ thống |
| `teencode – chuẩn hóa` | teencode hóa → **chuẩn hóa** (P0.2) |
| `teencode + không dấu – thô / xử lý` | teencode hóa rồi bỏ dấu – kiểu gõ cẩu thả thực tế nhất; bản xử lý = chuẩn hóa + khôi phục dấu |
""")
code(r"""
from vicare_safe.metrics import first_gold_rank, retrieval_summary, hits_at, bootstrap_ci, mcnemar_p
from vicare_safe.confidence import features

from vicare_safe.pipeline import preprocess_query
def process(q):          # đúng bước tiền xử lý của pipeline: teencode → khôi phục dấu
    return preprocess_query(q, restorer)

VARIANTS = {
    "có dấu":               [(q["question"], False, False) for q in dev],
    "không dấu – thô":      [(fold(q["question"]), False, False) for q in dev],
    "không dấu – khôi phục": [process(fold(q["question"])) for q in dev],
    "teencode – thô":       [(t["question"], False, False) for t in dev_teen],
    "teencode – chuẩn hóa": [process(t["question"]) for t in dev_teen],
    "teencode + không dấu – thô": [(fold(t["question"]), False, False) for t in dev_teen],
    "teencode + không dấu – xử lý": [process(fold(t["question"])) for t in dev_teen],
}

def run(items, gold_sets, method):
    rows, t0 = [], time.time()
    for (query, norm, rest), gold in zip(items, gold_sets):
        hits = retr.dense(query, k=10) if method == "dense" else retr.rerank(query, k=10, n_cand=N_CAND)
        rows.append({"query": query, "hits": hits, "rank": first_gold_rank([h["doc_id"] for h in hits], gold),
                     "feats": features(hits, query, rest, norm) if method != "dense" else None,
                     "top": hits[0]["score"] if hits else 0.0})
    return rows, (time.time() - t0) / max(len(items), 1)

gold_dev = [set(q["gold_doc_ids"]) for q in dev]
runs, latency = {}, {}
for name, items in VARIANTS.items():
    for method in ("dense", "v3"):
        if method == "dense" and name.endswith(("khôi phục", "chuẩn hóa", "xử lý")):
            continue
        runs[(method, name)], latency[(method, name)] = run(items, gold_dev, method)
        r = retrieval_summary([x["rank"] for x in runs[(method, name)]])
        print(f"{method:5} | {name:22} | " + " ".join(f"{k}={v:.2f}" for k, v in r.items()) + f" | {latency[(method, name)]*1000:.0f} ms/câu")
    reranker.save()
""")
code(r"""
table = pd.DataFrame({f"{'Baseline BTC (vector)' if m == 'dense' else 'ViCare-Safe v3'} | {n}":
                      retrieval_summary([x["rank"] for x in rows]) for (m, n), rows in runs.items()}).T
display(table.style.format("{:.1%}"))

print("Recall@5 – CI 95% và McNemar so với baseline cùng điều kiện gõ:")
significance = {}
for name in VARIANTS:
    base_name = name.replace(" – khôi phục", " – thô").replace(" – chuẩn hóa", " – thô").replace(" – xử lý", " – thô")
    b = hits_at([x["rank"] for x in runs[("dense", base_name)]], 5)
    v = hits_at([x["rank"] for x in runs[("v3", name)]], 5)
    lo, hi = bootstrap_ci(v); p, lose, win = mcnemar_p(b, v)
    print(f"  {name:22} baseline {np.mean(b):.0%} → v3 {np.mean(v):.0%} [CI {lo:.0%}–{hi:.0%}] | thắng {win} / thua {lose} | p={p:.4f}")
    significance[name] = {"baseline_R@5": float(np.mean(b)), "v3_R@5": float(np.mean(v)), "ci95": [lo, hi],
                          "wins": win, "losses": lose, "mcnemar_p": p}
""")

md(r"""
## 6. P0.3 – Độ tin cậy

1. **Cổng ngoài phạm vi (giữ từ v2):** điểm reranker top-1 < τ → từ chối. τ chọn trên tập tune để từ chối ≥ 95% câu ngoài phạm vi.
2. **Độ tin cậy:** hồi quy logistic trên 7 tín hiệu (điểm reranker top-1, khoảng cách top-1/top-2, trung bình top-3, BM25 và vector cùng xếp tài liệu vào top-5, cosine vector, cờ khôi phục dấu, độ dài câu).
   Nhãn = 1 nếu tài liệu gốc nằm trong **3 nguồn hiển thị**; câu ngoài phạm vi = 0. Huấn luyện **chỉ trên tập tune (train)**.
3. **Mức:** 🟢 Cao (≥ 70%) · 🟡 Trung bình (40–70%) · 🔴 Thấp (< 40% → không trả lời, chỉ hiện nguồn).
""")
code(r"""
from vicare_safe.confidence import ConfidenceModel, pick_oos_threshold, level_of, reliability_by_level, FEATURES
from vicare_safe.metrics import ece, brier, auroc

gold_tune = [set(q["gold_doc_ids"]) for q in tune]
tune_runs = {
    "có dấu": run([(q["question"], False, False) for q in tune], gold_tune, "v3")[0],
    "không dấu – khôi phục": run([process(fold(q["question"])) for q in tune], gold_tune, "v3")[0],
}
oos_tune = run([(q, False, False) for q in OOS_TUNE] + [process(fold(q)) for q in OOS_TUNE], [set()] * 30, "v3")[0]
reranker.save()

TAU = pick_oos_threshold([x["top"] for x in tune_runs["có dấu"]], [x["top"] for x in oos_tune], OOS_TARGET)
X = [x["feats"] for rows in tune_runs.values() for x in rows] + [x["feats"] for x in oos_tune]
y = [int(x["rank"] is not None and x["rank"] <= 3) for rows in tune_runs.values() for x in rows] + [0] * len(oos_tune)
conf = ConfidenceModel(l2=1.0).fit(X, y)
print(f"τ (cổng ngoài phạm vi) = {TAU:.3f} | mẫu huấn luyện: {len(y)} (dương {sum(y)})")
display(pd.Series(conf.coefficients(), name="hệ số (đặc trưng đã chuẩn hóa)").to_frame().style.format("{:+.2f}"))
""")
code(r"""
oos_test = run([(q, False, False) for q in OOS_TEST], [set()] * len(OOS_TEST), "v3")[0]
oos_test_nd = run([process(fold(q)) for q in OOS_TEST], [set()] * len(OOS_TEST), "v3")[0]
reranker.save()

def decide(rows):
    out = []
    for x in rows:
        if x["top"] < TAU:
            out.append(("Từ chối", None))
        else:
            p = conf.predict_one(x["feats"]); out.append((level_of(p), p))
    return out

dist, rel_rows, all_p, all_y = {}, [], [], []
for name in VARIANTS:
    rows = runs[("v3", name)]
    dec = decide(rows)
    labels = [int(x["rank"] is not None and x["rank"] <= 3) for x in rows]
    dist[name] = pd.Series([d for d, _ in dec]).value_counts(normalize=True)
    for (lvl, p), lab in zip(dec, labels):
        if p is not None:
            all_p.append(p); all_y.append(lab)
for name, rows in [("ngoài phạm vi (có dấu)", oos_test), ("ngoài phạm vi (không dấu)", oos_test_nd)]:
    dist[name] = pd.Series([d for d, _ in decide(rows)]).value_counts(normalize=True)

order = ["Cao", "Trung bình", "Thấp", "Từ chối"]
dist_df = pd.DataFrame(dist).T.reindex(columns=order).fillna(0)
print("Phân bố quyết định (tỷ lệ câu):")
display(dist_df.style.format("{:.0%}"))

rel = pd.DataFrame(reliability_by_level(all_p, all_y)).set_index("level")
rel.columns = ["số câu", "độ tin cậy dự đoán TB", "tỷ lệ tìm đúng thực tế"]
print("Độ tin cậy có ý nghĩa không? (gộp mọi biến thể dev, câu không bị từ chối)")
display(rel.style.format({"số câu": "{:.0f}", "độ tin cậy dự đoán TB": "{:.0%}", "tỷ lệ tìm đúng thực tế": "{:.0%}"}))
calib = {"ECE": ece(all_p, all_y), "Brier": brier(all_p, all_y),
         "AUROC": auroc([p for p, l in zip(all_p, all_y) if l], [p for p, l in zip(all_p, all_y) if not l])}
print({k: round(v, 3) for k, v in calib.items()})
""")

md(r"""
### So với v2: câu không dấu còn bị từ chối bao nhiêu?

v2 (một ngưỡng duy nhất, không khôi phục dấu) **từ chối nhầm 80% câu dev không dấu**. Ở v3, câu không dấu được khôi phục dấu trước khi chấm điểm.
""")
code(r"""
v2_style = {name: np.mean([x["top"] < TAU for x in runs[("v3", name)]]) for name in VARIANTS}
compare = pd.DataFrame({"bị từ chối (cổng)": v2_style,
                        "trả lời (Cao/TB)": {n: dist_df.loc[n, ["Cao", "Trung bình"]].sum() for n in VARIANTS},
                        "chỉ hiện nguồn (Thấp)": {n: dist_df.loc[n, "Thấp"] for n in VARIANTS}})
display(compare.style.format("{:.0%}"))
""")

md("## 7. Lớp an toàn (giữ từ v2) – kiểm tra lại")
code(r"""
from vicare_safe.safety import EMERG_POS, EMERG_NEG, DOSE_POS, is_emergency, is_emergency_baseline, is_dose_request
safety = {}
for name, fn in [("baseline BTC", is_emergency_baseline), ("ViCare-Safe", is_emergency)]:
    safety[name] = {"cấp cứu – có dấu": np.mean([fn(q) for q in EMERG_POS]),
                    "cấp cứu – không dấu": np.mean([fn(fold(q)) for q in EMERG_POS]),
                    "báo động giả": np.mean([fn(q) for q in EMERG_NEG] + [fn(fold(q)) for q in EMERG_NEG])}
safety["ViCare-Safe"]["chặn hỏi liều/kê đơn"] = np.mean([is_dose_request(q) for q in DOSE_POS])
display(pd.DataFrame(safety).style.format("{:.0%}", na_rep="–"))
""")

md("## 8. P0.4 – Đầu ra hỗ trợ quyết định (bằng chứng + độ tin cậy)")
code(r"""
from vicare_safe.pipeline import ViCareSafe
app = ViCareSafe(retr, restorer, conf, tau=TAU, n_cand=N_CAND)
EXAMPLES = [
    ("Câu chuẩn", dev[0]["question"], None),
    ("Không dấu + cấp cứu", "con toi 2 tuoi sot cao co giat phai lam sao", None),
    ("Teencode", "bs ơi bà bầu uống thuốc hạ sốt dc ko", None),
    ("Ngoài phạm vi", "Giá vàng hôm nay bao nhiêu?", None),
    ("Hỏi liều", "Trẻ 15kg uống paracetamol liều bao nhiêu?", None),
    ("Câu nối tiếp", "Vậy có cần đi khám chuyên khoa không?", [dev[1]["question"]]),
]
for label, q, hist in EXAMPLES:
    a = app.ask(q, hist)
    print("=" * 100); print(f"[{label}] ❓ {q}" + (f"   (lịch sử: {hist[-1][:60]}…)" if hist else "")); print("-" * 100)
    print(a.markdown)
reranker.save()
""")

md("## 9. Tổng hợp và lưu kết quả")
code(r"""
def r5(m, n): return retrieval_summary([x["rank"] for x in runs[(m, n)]])
summary = {
    "restoration": restoration,
    "retrieval": {f"{m}|{n}": r5(m, n) for (m, n) in runs},
    "decisions": {n: dist_df.loc[n].to_dict() for n in dist_df.index},
    "reliability": rel.reset_index().to_dict(orient="records"),
    "calibration": calib,
    "significance_R@5": significance,
    "safety": safety,
    "tau": TAU,
    "config": {"N_CAND": N_CAND, "RERANK_MAX_LEN": RERANK_MAX_LEN, "OOS_TARGET": OOS_TARGET, "N_TUNE": N_TUNE},
    "latency_ms": {f"{m}|{n}": v * 1000 for (m, n), v in latency.items()},
}
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
conf.save(ROOT / "artifacts/confidence.json", tau=TAU, n_cand=N_CAND, rerank_max_len=RERANK_MAX_LEN)

head = []
for label, base, new in [("có dấu", "có dấu", "có dấu"), ("không dấu", "không dấu – thô", "không dấu – khôi phục"),
                         ("teencode", "teencode – thô", "teencode – chuẩn hóa"),
                         ("teencode + không dấu", "teencode + không dấu – thô", "teencode + không dấu – xử lý")]:
    b, v = r5("dense", base), r5("v3", new)
    head.append({"Điều kiện gõ": label, "Baseline R@1": b["R@1"], "v3 R@1": v["R@1"],
                 "Baseline R@5": b["R@5"], "v3 R@5": v["R@5"], "Baseline MRR": b["MRR@10"], "v3 MRR": v["MRR@10"]})
display(pd.DataFrame(head).set_index("Điều kiện gõ").style.format("{:.1%}"))
print("✅ Đã lưu reports/metrics.json và artifacts/confidence.json")
""")

md(r"""
## 10. Hạn chế đã nhận diện

1. **Đánh giá trong kho đã biết:** câu dev có đáp án nằm sẵn trong kho; chưa đo khả năng tổng quát hóa sang tài liệu mới.
2. **Biến thể không dấu / teencode là tổng hợp** (dữ liệu gốc: 0% câu không dấu, 0,6% có teencode) – người dùng thật có thể gõ sai chính tả theo cách khác.
3. **Độ tin cậy** chỉ phản ánh việc *tìm đúng tài liệu*, không phải độ đúng y khoa; huấn luyện trên ~430 mẫu (câu train) nên còn thô, và chưa thấy câu teencode khi học.
4. **Gold duy nhất:** nhiều câu trả lời gần trùng (đặc biệt về vaccine Covid) nên Recall chặt có thể đánh giá thấp hệ thống.
5. **Bộ câu cấp cứu / ngoài phạm vi tự viết, cỡ nhỏ** → kết quả lạc quan; phát hiện cấp cứu dựa trên từ khóa.
6. **Dữ liệu cũ** (khoảng 2019–2022; 20% tài liệu thuộc giai đoạn Covid-19) và chỉ dùng cho mục đích nghiên cứu.
7. Chế độ trích xuất **không tóm tắt** – câu trả lời là câu nguyên văn của bác sĩ, có thể dài hoặc thiếu mạch lạc.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "ViCare_Safe_v3.ipynb"
nbf.write(nb, out)
print("wrote", out)
