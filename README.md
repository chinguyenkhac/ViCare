# ViCare-Safe 🩺

**Tra cứu sức khỏe tiếng Việt – hiểu cả khi bạn gõ không dấu, và biết nói “tôi chưa chắc”.**

[![CI](https://github.com/chinguyenkhac/ViCare/actions/workflows/ci.yml/badge.svg)](https://github.com/chinguyenkhac/ViCare/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chinguyenkhac/ViCare/blob/main/notebooks/ViCare_Safe_v3.ipynb)
[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-c2412d)](https://chinguyenkhac.github.io/ViCare/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Bài dự thi **AI Challenge 24h** sau workshop *AI Engineer Roadmap* (Viện AI4LIFE – ĐH Bách Khoa Hà Nội & Rikkei).
Phát triển từ notebook RAG **ViCare** của Ban tổ chức (dữ liệu ViHealthQA, embedding e5, Qwen2.5 qua Ollama).

> ⚠️ Sản phẩm minh họa học thuật – **không** chẩn đoán, **không** kê đơn, không thay thế nhân viên y tế. Cấp cứu: gọi **115**.

- 🌐 **Trang demo (ai cũng xem được):** https://chinguyenkhac.github.io/ViCare/ – chọn câu hỏi, đổi cách gõ có dấu / không dấu / teencode, xem đầu ra thật của hệ thống.
- 💬 **Demo chat trực tiếp:** Hugging Face Space – xem mục [Triển khai demo](#triển-khai-demo-hugging-face-space).
- 📓 **Notebook đánh giá đã chạy:** [`notebooks/ViCare_Safe_v3.ipynb`](notebooks/ViCare_Safe_v3.ipynb) · bản HTML: [`reports/ViCare_Safe_v3.html`](reports/ViCare_Safe_v3.html)

---

## 1. Bài toán & vì sao cần cải tiến

Phân tích notebook gốc trên **cùng 100 câu dev của BTC** cho thấy ba vấn đề thực tế:

| Vấn đề | Bằng chứng | Hậu quả với người dùng thật |
|---|---|---|
| Gõ **không dấu** là hỏng | Recall@5 của baseline chỉ còn **1%**; ở v2 (đã có hybrid + reranker) vẫn **từ chối nhầm 80%** câu không dấu | Người lớn tuổi, người đang vội gõ "con toi bi sot co giat" gần như không nhận được gì |
| **Cảnh báo cấp cứu** bỏ sót | Từ khóa có dấu không khớp câu không dấu: bắt được **0/20** ca cấp cứu gõ không dấu | Lỗi an toàn, không chỉ là lỗi chất lượng |
| Chỉ có **trả lời / từ chối** | Một ngưỡng cosine cứng, không nói được "tôi chưa chắc" | Người dùng không biết khi nào nên tin, khi nào nên hỏi bác sĩ |

## 2. Giải pháp

ViCare-Safe là hệ **hỗ trợ quyết định**: thay vì để LLM tự viết câu trả lời, hệ thống **trích nguyên văn lời bác sĩ** trong kho tài liệu,
kèm **độ tin cậy**, **bằng chứng [n]**, **nguồn** và **bước tiếp theo** – chạy được hoàn toàn trên CPU, không thể "bịa".

```text
câu hỏi (+ lịch sử)
  ├─ [an toàn]     cấp cứu (có dấu & không dấu) → banner 115 · hỏi liều / kê đơn → không đưa liều
  ├─ [chuẩn hóa]   teencode "ko, j, dc, bs" → dạng chuẩn · không dấu → KHÔI PHỤC DẤU (n-gram + Viterbi)
  ├─ [ngữ cảnh]    câu nối tiếp ("Vậy…") → ghép với câu trước
  ├─ [truy xuất]   BM25 (bỏ dấu, bigram âm tiết) + vector e5 → RRF → reranker bge-reranker-v2-m3
  ├─ [cổng]        điểm reranker < τ → từ chối (câu ngoài phạm vi)
  ├─ [độ tin cậy]  hồi quy logistic trên 8 tín hiệu → 🟢 Cao · 🟡 Trung bình · 🔴 Thấp (không trả lời, chỉ hiện nguồn)
  ├─ [bằng chứng]  chấm từng câu trong tài liệu → trích câu liên quan nhất kèm [n]
  └─ ĐẦU RA        câu hỏi đã hiểu · độ tin cậy · thông tin tham khảo [n] · bước tiếp theo · nguồn · cảnh báo
```

| Cải tiến | Cách làm | Vì sao chọn cách này |
|---|---|---|
| **Khôi phục dấu** | Học unigram/bigram âm tiết có dấu từ chính kho tài liệu + câu hỏi tập train, giải mã Viterbi | Không cần GPU/LLM, < 1 ms/câu, đúng văn phong y tế; giữ nguyên âm tiết người dùng đã gõ dấu |
| **Chuẩn hóa teencode** | Từ điển có ranh giới từ + luật chống nhầm | Tránh lỗi kiểu "vitamin K" → "vitamin không", "ng/mL" → "người/mL", "viêm gan C" |
| **Độ tin cậy** | Logistic regression (numpy) trên điểm reranker, khoảng cách top-1/top-2, đồng thuận BM25–vector, cờ khôi phục dấu… | Nhỏ, minh bạch, kiểm chứng được bằng bảng "khi báo Cao thì đúng bao nhiêu %" |
| **Bằng chứng** | Cross-encoder chấm từng câu, bỏ câu chào hỏi | Người dùng tự kiểm chứng được – đúng tinh thần hỗ trợ quyết định |

**Độ tin cậy nghĩa là gì?** Xác suất ước lượng rằng *tài liệu gốc trả lời câu hỏi nằm trong 3 nguồn hiển thị* – độ tin vào việc
**tìm đúng tài liệu**, **không phải** độ đúng y khoa.

## 3. Kết quả

<!-- RESULTS -->

## 4. Cấu trúc dự án

```text
vicare_safe/
  text.py         bỏ dấu, tách câu, chuẩn hóa teencode, ẩn thông tin cá nhân
  diacritics.py   khôi phục dấu (n-gram âm tiết + Viterbi)
  data.py         nạp ViHealthQA, dựng kho / dev / tune giống hệt notebook BTC
  retrieval.py    vector (numpy) · BM25 (ma trận thưa) · RRF · reranker có cache
  confidence.py   đặc trưng + hồi quy logistic + mức độ tin cậy
  evidence.py     trích câu bằng chứng
  safety.py       phát hiện cấp cứu, chặn hỏi liều
  pipeline.py     ViCareSafe.ask() → đầu ra hỗ trợ quyết định
  metrics.py      Recall@K, MRR, bootstrap CI, McNemar, AUROC, ECE, Brier
app.py            demo Gradio (Hugging Face Space)
notebooks/        notebook đánh giá v3 (đã chạy)
reports/          metrics.json, examples.json, bản HTML của notebook
artifacts/        embedding e5 tính sẵn (float16), mô hình độ tin cậy
docs/             trang GitHub Pages
scripts/          build_embeddings · build_notebook · make_examples · build_site · smoke_data · deploy_space
tests/            pytest (chạy trên GitHub Actions cùng smoke test dữ liệu thật)
```

## 5. Chạy thử

**Colab (khuyên dùng):** bấm nút *Open in Colab* ở đầu trang → `Runtime → Run all`.

**Máy cá nhân:**

```bash
git clone https://github.com/chinguyenkhac/ViCare && cd ViCare
```

```bash
pip install -r requirements.txt pytest
```

```bash
pytest -q
```

```bash
python app.py
```

`app.py` mở demo chat tại `http://127.0.0.1:7860` (lần đầu tải dữ liệu và model mất vài phút).

## Triển khai demo (Hugging Face Space)

Workflow [`deploy-space.yml`](.github/workflows/deploy-space.yml) tự đẩy demo lên Space mỗi khi `main` thay đổi – chỉ cần thêm token **một lần**:

1. Tạo token **Write** tại huggingface.co → *Settings* → *Access Tokens*.
2. Trên GitHub: *Settings* → *Secrets and variables* → *Actions* → *New repository secret*, tên `HF_TOKEN`.
3. Vào tab *Actions* → *Deploy Hugging Face Space* → *Run workflow*.

Space sẽ có tại `https://huggingface.co/spaces/<tên-tài-khoản>/ViCare-Safe` (CPU miễn phí, ~5–15 giây mỗi câu); trang GitHub Pages tự gắn link.

## 6. Hạn chế

1. **Đánh giá trong kho đã biết**: câu dev có đáp án nằm sẵn trong kho; chưa đo tổng quát hóa sang tài liệu mới.
2. **Câu không dấu / teencode là tổng hợp** từ câu gốc (dữ liệu gốc: 0% câu không dấu, 0,6% có teencode).
3. **Độ tin cậy** huấn luyện trên ~230 mẫu và chỉ phản ánh việc tìm đúng tài liệu.
4. **Gold duy nhất**: nhiều câu trả lời gần trùng (vaccine Covid…) nên Recall chặt có thể đánh giá thấp hệ thống.
5. **Bộ câu cấp cứu / ngoài phạm vi tự viết, cỡ nhỏ** → kết quả an toàn lạc quan; phát hiện cấp cứu dựa trên từ khóa.
6. **Dữ liệu cũ** (khoảng 2019–2022, ~20% tài liệu thuộc giai đoạn Covid-19).
7. Chế độ trích xuất không tóm tắt; câu trích có thể dài.

## 7. Dữ liệu & trích dẫn

ViHealthQA – *chỉ dùng cho mục đích nghiên cứu* (theo thẻ dữ liệu). Repo **không** phân phối lại dữ liệu; dữ liệu được tải từ
[Hugging Face](https://huggingface.co/datasets/tarudesu/ViHealthQA) khi chạy. Thông tin cá nhân dạng email / số điện thoại được ẩn.

> Nguyen, N. T.-H., Ha, P. P.-D., Nguyen, L. T., Van Nguyen, K., & Nguyen, N. L.-T. (2022).
> *SPBERTQA: A Two-Stage Question Answering System Based on Sentence Transformers for Medical Texts.* KSEM 2022.

Mã nguồn: [MIT](LICENSE). Tiến độ & quyết định thiết kế: [`progress.md`](progress.md).

---

<details>
<summary><b>English summary</b></summary>

ViCare-Safe is a Vietnamese health-information RAG assistant built as a decision-support tool. It quotes doctors' answers
verbatim from the ViHealthQA corpus with a calibrated confidence level, evidence sentences and sources, instead of letting an
LLM generate advice. P0 improvements over the organizers' baseline: corpus-trained n-gram/Viterbi **diacritic restoration**
(Vietnamese typed without accents), **teencode normalization**, a logistic **confidence model** (High / Medium / Low; Low shows
sources only), and **evidence extraction** with a cross-encoder. Everything runs on CPU; unit tests and a real-data smoke test
run on GitHub Actions; the evaluation notebook reports Recall@K, MRR, bootstrap CIs, McNemar tests, ECE and AUROC.
For academic demonstration only – not medical advice.
</details>
