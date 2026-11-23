"""ViCare-Safe – demo Gradio (Hugging Face Space, CPU miễn phí, chế độ trích xuất không LLM)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np

from vicare_safe.confidence import ConfidenceModel, norm_reliability
from vicare_safe.data import load_kb
from vicare_safe.pipeline import ViCareSafe, build_restorer
from vicare_safe.retrieval import HybridRetriever, Reranker, load_embedder

ROOT = Path(__file__).resolve().parent
GITHUB = "https://github.com/chinguyenkhac/ViCare"
COLAB = "https://colab.research.google.com/github/chinguyenkhac/ViCare/blob/main/notebooks/ViCare_Safe_v3.ipynb"

try:
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except ImportError:  # pragma: no cover
    pass


def build_app() -> ViCareSafe:
    t0 = time.time()
    kb = load_kb()
    emb = np.load(ROOT / "artifacts/e5_small_chunks.f16.npy").astype(np.float32)
    ids = json.loads((ROOT / "artifacts/chunk_ids.json").read_text())["ids"]
    if ids != [c["id"] for c in kb.chunks]:
        raise RuntimeError("Embedding tính sẵn không khớp kho tài liệu – chạy lại scripts/build_embeddings.py")
    conf, meta = ConfidenceModel.load(ROOT / "artifacts/confidence.json")
    retr = HybridRetriever(kb.chunks, emb, load_embedder(),
                           Reranker(max_length=int(meta.get("rerank_max_len", 320))))
    app = ViCareSafe(retr, build_restorer(kb), conf, tau=float(meta["tau"]), n_cand=int(meta.get("n_cand", 10)))
    print(f"ViCare-Safe sẵn sàng sau {time.time() - t0:.0f}s", flush=True)
    return app


APP = build_app()


def _user_turns(history) -> list[str]:
    turns = []
    for item in history or []:
        if isinstance(item, dict):
            if item.get("role") == "user":
                c = item.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                if isinstance(c, str) and c.strip():
                    turns.append(c)
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            turns.append(item[0])
    return turns


def chat(message, history):
    t0 = time.time()
    a = APP.ask(str(message), _user_turns(history))
    return a.markdown + f"\n\n<sub>⏱ {time.time() - t0:.1f}s · chế độ trích xuất (không LLM)</sub>"


# ----------------------------------------------------------------------------- tab kết quả
def _pct(x):
    return "–" if x is None else f"{x:.0%}"


def results_markdown() -> str:
    p = ROOT / "reports/metrics.json"
    if not p.exists():
        return "Chưa có kết quả đánh giá. Chạy notebook `notebooks/ViCare_Safe_v3.ipynb`."
    m = json.loads(p.read_text())
    r = m["retrieval"]
    rows = []
    for label, base, new in [("Có dấu", "dense|có dấu", "v3|có dấu"),
                             ("Không dấu", "dense|không dấu – thô", "v3|không dấu – khôi phục"),
                             ("Teencode", "dense|teencode – thô", "v3|teencode – chuẩn hóa"),
             ("Teencode + không dấu", "dense|teencode + không dấu – thô", "v3|teencode + không dấu – xử lý")]:
        b, v = r[base], r[new]
        rows.append(f"| {label} | {_pct(b['R@1'])} → **{_pct(v['R@1'])}** | {_pct(b['R@5'])} → **{_pct(v['R@5'])}** "
                    f"| {_pct(b['MRR@10'])} → **{_pct(v['MRR@10'])}** |")
    rel = "\n".join(f"| {x['level']} | {x['n']:.0f} | {_pct(x['mean_conf'])} | {_pct(x['actual_acc'])} |"
                    for x in norm_reliability(m["reliability"]))
    dec = "\n".join(f"| {k} | " + " | ".join(_pct(v.get(c, 0.0)) for c in ["Cao", "Trung bình", "Thấp", "Từ chối"]) + " |"
                    for k, v in m["decisions"].items())
    s = m["safety"]
    rest = m["restoration"]
    return f"""
### Truy xuất: baseline BTC → ViCare-Safe v3 (100 câu dev của BTC)
| Điều kiện gõ | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
{chr(10).join(rows)}

**Khôi phục dấu:** {rest['syllable_acc']:.1%} âm tiết đúng · {rest['sentence_exact']:.0%} câu đúng hoàn toàn (100 câu dev, không dùng để học).

### Độ tin cậy có ý nghĩa không?
| Mức | Số câu | Độ tin cậy dự đoán TB | Tỷ lệ tìm đúng thực tế |
|---|---|---|---|
{rel}

ECE = {m['calibration']['ECE']:.3f} · AUROC = {m['calibration']['AUROC']:.2f} · Brier = {m['calibration']['Brier']:.3f}

### Phân bố quyết định
| Bộ câu hỏi | Cao | Trung bình | Thấp | Từ chối |
|---|---|---|---|---|
{dec}

### An toàn
| | Baseline BTC | ViCare-Safe |
|---|---|---|
| Cấp cứu – có dấu | {_pct(s['baseline BTC']['cấp cứu – có dấu'])} | {_pct(s['ViCare-Safe']['cấp cứu – có dấu'])} |
| Cấp cứu – không dấu | {_pct(s['baseline BTC']['cấp cứu – không dấu'])} | {_pct(s['ViCare-Safe']['cấp cứu – không dấu'])} |
| Báo động giả | {_pct(s['baseline BTC']['báo động giả'])} | {_pct(s['ViCare-Safe']['báo động giả'])} |

*Chi tiết, khoảng tin cậy và kiểm định McNemar: [notebook]({COLAB}).*
"""


ABOUT = f"""
### ViCare-Safe là gì?
Trợ lý **tra cứu** sức khỏe tiếng Việt theo kiểu **hỗ trợ quyết định**: không chẩn đoán, không kê đơn – chỉ đưa ra
**câu nguyên văn của bác sĩ** trong kho tài liệu, kèm **độ tin cậy**, **nguồn** và **bước tiếp theo**.

### Luồng xử lý
```
câu hỏi → an toàn (cấp cứu 115 / chặn hỏi liều) → chuẩn hóa teencode → khôi phục dấu
        → BM25 (bỏ dấu) + vector e5 → RRF → reranker → cổng ngoài phạm vi
        → độ tin cậy (logistic) → trích đoạn bằng chứng → đầu ra hỗ trợ quyết định
```

### Độ tin cậy nghĩa là gì?
Xác suất ước lượng rằng **tài liệu gốc trả lời câu hỏi nằm trong 3 nguồn hiển thị** – tức độ tin vào việc *tìm đúng
tài liệu*, **không phải** độ đúng y khoa. 🟢 Cao ≥ 70% · 🟡 Trung bình 40–70% · 🔴 Thấp < 40% (không trả lời, chỉ hiện nguồn).

### Hạn chế
- Dữ liệu ViHealthQA (Vinmec, VnExpress) thu thập khoảng 2019–2022, chỉ dùng cho mục đích nghiên cứu; có thể đã cũ.
- Chế độ trích xuất không tóm tắt; câu trích có thể dài.
- Phát hiện cấp cứu dựa trên từ khóa – **luôn gọi 115 khi nghi ngờ cấp cứu**.

### Nguồn
- Mã nguồn: [{GITHUB}]({GITHUB}) · Notebook đánh giá: [Colab]({COLAB})
- Dữ liệu: *Nguyen et al., SPBERTQA: A Two-Stage Question Answering System Based on Sentence Transformers for Medical
  Texts, KSEM 2022* – [tarudesu/ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA)
"""

EXAMPLES = [
    "con toi 2 tuoi sot cao co giat phai lam sao",
    "bs ơi bà bầu uống thuốc hạ sốt dc ko",
    "Đang chích ngừa viêm gan B có chích ngừa Covid-19 được không?",
    "Giá vàng hôm nay bao nhiêu?",
    "Trẻ 15kg uống paracetamol liều bao nhiêu?",
]

with gr.Blocks(title="ViCare-Safe") as demo:
    gr.Markdown(
        "# 🩺 ViCare-Safe – Tra cứu sức khỏe tiếng Việt có độ tin cậy\n"
        "Hiểu cả câu **không dấu** và **teencode** · trích **nguyên văn** lời bác sĩ kèm **bằng chứng** & **nguồn** · "
        "**không chẩn đoán, không kê đơn**. Cấp cứu: gọi **115**.\n\n"
        f"[GitHub]({GITHUB}) · [Notebook đánh giá]({COLAB}) · Dùng GPU nếu có (nhanh hơn nhiều); trên CPU mỗi câu mất ~10–60 giây."
    )
    with gr.Tab("💬 Hỏi đáp"):
        gr.ChatInterface(fn=chat, examples=EXAMPLES, cache_examples=False,
                         description="Không nhập thông tin cá nhân. Câu hỏi nối tiếp (\"Vậy…\") được ghép với câu trước.")
    with gr.Tab("📊 Kết quả đánh giá"):
        gr.Markdown(results_markdown())
    with gr.Tab("ℹ️ Giới thiệu"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(theme=gr.themes.Soft())
