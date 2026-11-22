# ViCare-Safe – Tiến độ triển khai (P0)

> Tài liệu sống: cập nhật trạng thái sau mỗi mốc. Hạn nộp AI Challenge: **12:00 ngày 23/11/2026 (giờ VN)**.
> Quỹ thời gian cho đợt này: **~3 giờ** (13:00 → 16:00 giờ VN, 22/11/2026).

## 1. Mục tiêu đợt này

Biến notebook ViCare-Safe v2 thành **một dự án GitHub hoàn chỉnh**: code tách module có test, CI chạy trên GitHub,
notebook đánh giá đã chạy (kèm bản HTML) và **demo công khai trên Hugging Face Space**. Đợt này chỉ làm **P0**.

## 2. Quy mô

### Trong phạm vi (P0)

| # | Hạng mục | Kết quả mong đợi |
|---|---|---|
| P0.1 | **Khôi phục dấu tiếng Việt** (n-gram thống kê từ chính kho tài liệu, giải mã Viterbi) | Câu không dấu → câu có dấu trước khi tìm kiếm & rerank |
| P0.2 | **Chuẩn hóa teencode / viết tắt** | "ko", "j", "dc", "bs", "bv"… → dạng chuẩn |
| P0.3 | **Độ tin cậy (confidence)** | Xác suất "tài liệu gốc nằm trong 3 nguồn hiển thị", chia mức Cao / Trung bình / Thấp; Thấp → không tổng hợp câu trả lời |
| P0.4 | **Trích đoạn bằng chứng** | 1–3 câu nguyên văn trong tài liệu bác sĩ làm căn cứ, kèm số nguồn [n] |
| P0.5 | **Repo GitHub + CI** | Module `vicare_safe/`, pytest, GitHub Actions (unit test + smoke test trên dữ liệu thật), nút Open in Colab |
| P0.6 | **Demo Hugging Face Space** | Link công khai, cố định; chế độ trích xuất (không LLM) chạy được trên CPU miễn phí |
| — | **Giữ nguyên từ v2** | Hybrid BM25 + vector + reranker, cổng từ chối câu ngoài phạm vi, phát hiện cấp cứu v2, chặn hỏi liều/kê đơn, nhớ hội thoại (ghép câu) |

### Ngoài phạm vi đợt này (để P1/P2 nếu còn thời gian)

- P1: Đối chiếu đối tượng (trẻ em / thai kỳ), Chế độ thử cho giám khảo, Gợi ý câu hỏi tiếp + hỏi lại khi mơ hồ, Gợi ý chuyên khoa, Cảnh báo dữ liệu cũ, 👍/👎, Recall nới (near-duplicate).
- P2: Giải nghĩa thuật ngữ, Kiểm tra một câu (NLI + chặn thực thể), Router LLM, GitHub Pages.

## 3. Quyết định đã chốt (từ trao đổi)

1. **Demo công khai dùng chế độ trích xuất** (trích nguyên văn câu bác sĩ + bằng chứng, không LLM): nhanh trên CPU, không thể bịa. LLM tóm tắt để ở Colab (tùy chọn).
2. **Độ tin cậy Thấp** → không tổng hợp câu trả lời; chỉ hiện nguồn liên quan + đoạn bằng chứng + lời khuyên hỏi nhân viên y tế. Câu ngoài phạm vi → từ chối như v2.
3. **Notebook đã chạy**: chạy phần không cần LLM trên máy CPU của phiên làm việc; phần LLM giữ số liệu v2 (có thể chạy lại trên Colab GPU).
4. **Repo public**; dữ liệu **không** đẩy lên repo (tải từ Hugging Face khi chạy), chỉ lưu vector embedding dẫn xuất.
5. Dữ liệu: ViHealthQA (`tarudesu/ViHealthQA`) – *chỉ dùng cho mục đích nghiên cứu*, có trích dẫn bài báo gốc.

## 4. Ràng buộc kỹ thuật đã đo

| Đo trên CPU 4 nhân của phiên | Giá trị |
|---|---|
| Reranker `bge-reranker-v2-m3`, 10 cặp | ~3,3 s (4 nhân) / ~5,8 s (2 nhân) |
| Reranker, 20 cặp | ~11,9 s (4 nhân) / ~22,4 s (2 nhân) |
| Embedding `multilingual-e5-small` | ~7 tài liệu/s → tính sẵn 1 lần, lưu fp16 |

→ Space miễn phí (2 vCPU) dùng **N_CAND = 10** ứng viên cho reranker; đánh giá dùng cùng cấu hình để số liệu khớp với demo.

## 5. Kiến trúc

```text
câu hỏi (+ lịch sử)
  ├─ [an toàn]   cấp cứu? → banner 115 | hỏi liều/kê đơn? → không đưa liều
  ├─ [chuẩn hóa] teencode → dạng chuẩn ; không dấu → khôi phục dấu (n-gram Viterbi)
  ├─ [ngữ cảnh]  câu nối tiếp → ghép với câu trước
  ├─ [truy xuất] BM25 (bỏ dấu, bigram âm tiết) + vector e5 → RRF → reranker (top-10)
  ├─ [cổng]      điểm reranker < τ (chọn trên tập tune, từ chối ≥95% câu ngoài phạm vi) → từ chối
  ├─ [tin cậy]   hồi quy logistic trên tín hiệu truy xuất → Cao / Trung bình / Thấp
  ├─ [bằng chứng] chấm từng câu trong top tài liệu bằng reranker → 1–3 câu căn cứ
  └─ ĐẦU RA (hỗ trợ quyết định): câu hỏi đã hiểu | độ tin cậy | bằng chứng [n] | bước tiếp theo | nguồn | cảnh báo
```

## 6. Thiết kế chi tiết & tiêu chí nghiệm thu

### P0.1 Khôi phục dấu
- Mô hình: đếm unigram/bigram âm tiết có dấu từ kho tài liệu (câu trả lời) + câu hỏi tập train; ứng viên cho mỗi âm tiết bỏ dấu = các dạng có dấu cùng "khung" (top theo tần suất); giải mã Viterbi với nội suy bigram/unigram.
- Âm tiết đã có dấu giữ nguyên (xử lý được câu gõ lẫn có dấu / không dấu).
- **Đo:** % âm tiết khôi phục đúng & % câu đúng hoàn toàn trên 100 câu dev; Recall@k và tỷ lệ từ chối câu không dấu **trước/sau**.
- **Nghiệm thu:** độ chính xác âm tiết ≥ 85%; tỷ lệ từ chối nhầm câu không dấu giảm rõ so với v2 (80%).

### P0.2 Teencode
- Từ điển chuẩn hóa có ranh giới từ (tránh thay nhầm trong từ khác). Chỉ dùng các viết tắt ít nhập nhằng.
- Bộ test: **dev "teencode hóa" tổng hợp** (dữ liệu gốc gần như không có teencode – 0,6% câu hỏi).
- **Đo:** Recall@k trên dev teencode trước/sau chuẩn hóa.

### P0.3 Độ tin cậy
- Nhãn huấn luyện: `1` nếu tài liệu gốc nằm trong **3 nguồn đầu** (đúng những gì người dùng nhìn thấy); câu ngoài phạm vi = `0`.
- Đặc trưng (7): điểm reranker top-1, khoảng cách top-1/top-2, trung bình top-3, độ đồng thuận BM25–vector, cosine vector top-1, cờ đã khôi phục dấu, độ dài câu. (Bỏ cờ teencode – xem nhật ký 14:20.)
- Mô hình: hồi quy logistic (numpy), huấn luyện trên **tập tune (200 câu train + bản không dấu + 30 câu ngoài phạm vi)**, đánh giá trên **dev + biến thể + OOS_TEST**.
- Mức: Cao (p ≥ 0,7) · Trung bình (0,4 ≤ p < 0,7) · Thấp (p < 0,4).
- **Đo:** độ chính xác thực tế theo từng mức ("khi báo Cao thì đúng X%"), ECE, Brier, AUROC.
- **Lưu ý trình bày:** độ tin cậy = độ tin *tài liệu khớp câu hỏi*, không phải độ đúng y khoa.

### P0.4 Bằng chứng
- Tách câu (xử lý cả lỗi dính dấu chấm "…khoa.Em…"), bỏ câu chào hỏi/xã giao, reranker chấm từng câu với câu hỏi, chọn tốt nhất mỗi nguồn.
- **Đo:** định tính qua ví dụ trong notebook và demo.

### P0.5 GitHub + CI
- `pytest` cho các hàm thuần (bỏ dấu, teencode, khôi phục dấu, an toàn, BM25, RRF, logistic, metrics, định dạng đầu ra).
- Workflow: unit test + **smoke test trên dữ liệu thật** (tải ViHealthQA, đo độ chính xác khôi phục dấu và Recall BM25, không cần GPU).

### P0.6 Hugging Face Space
- Gradio, CPU basic; nạp embedding tính sẵn; tab **Hỏi đáp** và **Kết quả đánh giá**.

## 7. Kế hoạch đánh giá (notebook `notebooks/ViCare_Safe_v3.ipynb`)

| Bộ | Nguồn | Dùng cho |
|---|---|---|
| `dev` (100) | validation, seed 2026 – **giống BTC** | so sánh chính |
| `dev_nodau` | dev bỏ dấu | P0.1 |
| `dev_teencode` | dev teencode hóa (tổng hợp) | P0.2 |
| `tune` (100) | train, seed 7 | chọn τ, huấn luyện độ tin cậy |
| `OOS_TUNE` / `OOS_TEST` | 15 / 15 câu ngoài phạm vi (từ v2) | cổng từ chối, độ tin cậy |

Thống kê: CI 95% bootstrap, McNemar cho so sánh cặp.

## 8. Cấu trúc repo

```text
vicare_safe/      text.py · diacritics.py · data.py · safety.py · retrieval.py · evidence.py · confidence.py · pipeline.py · metrics.py
app.py            ứng dụng Gradio (Hugging Face Space)
scripts/          build_embeddings.py · smoke_data.py · build_notebook.py
notebooks/        ViCare_Safe_v3.ipynb (đã chạy)
reports/          metrics.json · ViCare_Safe_v3.html
artifacts/        embedding fp16 của kho tài liệu
tests/            pytest
.github/workflows/ci.yml
```

## 9. Checklist & trạng thái

- [x] Viết progress.md, dựng khung repo
- [x] `text.py`, `data.py` + tính embedding kho tài liệu (chạy nền)
- [x] P0.1 `diacritics.py` + test
- [x] P0.2 teencode + test
- [x] `retrieval.py` (vector numpy, BM25 sparse, RRF, reranker có cache) + test
- [x] P0.4 `evidence.py` + test
- [x] P0.3 `confidence.py` + test
- [x] `pipeline.py` (đầu ra hỗ trợ quyết định) + `safety.py` + test
- [x] CI GitHub Actions
- [x] Notebook v3: chạy đánh giá, xuất HTML, `reports/metrics.json`
- [x] `app.py` (đã test giao diện Gradio 6 qua HTTP)
- [ ] Deploy Hugging Face Space – **chờ token** (form bị hủy; đã làm sẵn workflow tự deploy khi có secret `HF_TOKEN`)
- [x] Trang demo tĩnh "Thử như giám khảo" + báo cáo HTML (`docs/`) – **bạn cần bật GitHub Pages 1 lần** (xem mục 13)
- [x] README (kết quả, cách chạy, kiến trúc, hạn chế, trích dẫn)
- [x] Cập nhật progress.md lần cuối

## 10. Rủi ro & phương án dự phòng

| Rủi ro | Phương án |
|---|---|
| CPU chậm → đánh giá không kịp | N_CAND=10, cache điểm reranker, chạy nền song song khi viết code |
| Không đẩy được workflow lên GitHub (thiếu quyền) | Giữ file workflow, hướng dẫn thêm thủ công |
| Space build lâu / thiếu RAM | Dùng embedding tính sẵn, nạp model lười, giới hạn ứng viên |
| Không có token Hugging Face | Chuẩn bị sẵn thư mục Space để bạn tự tải lên |

## 11. Việc cần bạn làm

1. Nhập **access token Hugging Face (quyền Write)** qua form bảo mật khi được hỏi – không dán vào chat.
2. Sau khi xong: xem demo, quay video / chụp ảnh, điền bài đăng.

## 12. Kết quả cuối (100 câu dev của BTC, CPU, reranker 10 ứng viên)

| Cách gõ | Recall@5 baseline → v3 | Bị từ chối (v2-style → v3) |
|---|---|---|
| Có dấu | 50% → **57%** (p = 0,09) | 18% → 18% |
| Không dấu | 3% → **54%** (p < 0,0001) | 80% → **21%** |
| Teencode + không dấu | 3% → **54%** (p < 0,0001) | 81% → **21%** |

- Khôi phục dấu (trigram): 97,5% âm tiết, 71% câu đúng hoàn toàn (dev); 97,9% / 74,8% trên 400 câu validation ngoài dev.
- Độ tin cậy theo mức (dự đoán / thực tế): Cao 83% / 79% · Trung bình 57% / 61% · Thấp 31% / 26%; ECE 0,119, AUROC 0,78.
- Câu ngoài phạm vi: từ chối 100%. Cấp cứu (bộ tự viết): 100% cả có dấu lẫn không dấu, 0 báo động giả.

## 13. Việc cần bạn làm (khoảng 5 phút)

1. **Bật trang demo công khai (GitHub Pages):** repo → *Settings* → *Pages* → *Build and deployment* → *Source*: **Deploy from a branch** →
   Branch **main**, thư mục **/docs** → *Save*. Sau ~1 phút: https://chinguyenkhac.github.io/ViCare/
2. **Demo chat trực tiếp (Hugging Face Space):** tạo token **Write** tại huggingface.co → *Settings* → *Access Tokens*; trên GitHub repo →
   *Settings* → *Secrets and variables* → *Actions* → *New repository secret* tên `HF_TOKEN`; rồi tab *Actions* →
   *Deploy Hugging Face Space* → *Run workflow*. Space: `https://huggingface.co/spaces/<tên-tài-khoản>/ViCare-Safe`
   (lần đầu build ~5–10 phút; trang Pages tự gắn link).
3. (Tùy chọn) Chia sẻ trang demo trên claude.ai (nút *Share*) nếu muốn gửi link ngay.
4. Quay video / chụp ảnh, điền bài đăng.

## 14. Nhật ký

- 13:02 – Chốt P0, viết progress.md.
- 13:11 – Embedding 11.227 chunk xong (320 s, CPU) → `artifacts/` (float16, 8,6 MB).
- 13:12 – Khôi phục dấu: **97,4% âm tiết đúng** trên 100 câu dev (0,6 ms/câu). 28 unit test qua.
- 13:13 – BM25 viết lại bằng ma trận thưa khớp v2 (R@5 43%, R@10 51%); smoke test dữ liệu thật 11 s.
- 13:16 – Push lần đầu lên `main`; **CI xanh** trên GitHub (unit test + smoke test dữ liệu thật).
- 13:17 – Form token Hugging Face bị hủy (bạn đi ngủ) → không tự mở lại; chuyển sang deploy qua GitHub Actions + trang GitHub Pages.
- 13:20 – Test giao diện Gradio 6 bằng pipeline giả lập: OK.
- 13:22 – API bật GitHub Pages bị proxy chặn → cần bật tay 1 lần (Settings → Pages → `main` / `/docs`). Trang demo tĩnh sẽ xuất bản thêm dạng Artifact trên claude.ai để xem được ngay.
- 13:25 – Thiết kế lại trang demo (phong cách phiếu xét nghiệm, màu phân loại cho độ tin cậy, sáng/tối, điện thoại không tràn ngang); sửa định dạng danh sách trích dẫn/nguồn.
- 14:09 – Notebook đánh giá chạy xong lần 1 (57 phút trên CPU). Phát hiện **lỗi độ tin cậy**: cờ "đã chuẩn hóa teencode" luôn = 0 khi huấn luyện → độ lệch chuẩn ≈ 0 → lúc dự đoán bị phóng đại, kéo 65% câu teencode xuống mức Thấp. Đã sửa (sd = 1 cho đặc trưng hằng) + thêm test.
- 14:12 – Nâng khôi phục dấu lên **trigram** (Viterbi bậc 2): trên 400 câu validation ngoài dev, câu đúng hoàn toàn 71,3% → 74,8%; sửa được "con toi 2 tuoi" → "con tôi 2 tuổi". Chạy lại notebook (điểm reranker đã cache).
- 14:20 – Lỗi teencode chưa hết: vài câu train bỏ dấu tình cờ có token như "dk" nên cờ teencode không hằng tuyệt đối → vẫn bị phóng đại. Bỏ hẳn đặc trưng này (không có dữ liệu học). Tăng tập tune lên 200 câu để hiệu chỉnh độ tin cậy tốt hơn; chạy lại notebook.
- 14:45 – Chạy pipeline thật trên bộ ví dụ lộ thêm lỗi: câu **teencode + không dấu** bị từ chối 4/5 vì chuẩn hóa teencode chèn chữ có dấu ('lm j' → 'làm gì') nên bước khôi phục dấu bị bỏ qua. Sửa: quyết định khôi phục theo câu gốc, ngưỡng ≤ 25% âm tiết có dấu (câu thật có dấu luôn ≥ 30%); gom tiền xử lý về một hàm dùng chung cho notebook và demo; thêm biến thể 'teencode + không dấu' vào đánh giá. Ví dụ 'đau dạ dày' bị cổng từ chối nhầm → giữ lại trên trang demo với nhãn 'giới hạn đã biết'.
- 14:55 – Lần chạy cuối (biến thể teencode + không dấu): R@5 3% → 54%, từ chối 81% → 21%; độ tin cậy ECE 0,119. Dựng trang demo (Pages + Artifact claude.ai), bấm thử OK, không lỗi JS, không tràn ngang.
