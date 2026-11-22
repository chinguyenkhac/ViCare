"""Điền bảng kết quả vào README (giữa hai dấu RESULTS) từ reports/metrics.json."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vicare_safe.confidence import norm_reliability  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def pct(x):
    return "–" if x is None else f"{x * 100:.0f}%"


def render(m: dict) -> str:
    r = m["retrieval"]
    rows = []
    for name, b, v in [("Có dấu", "dense|có dấu", "v3|có dấu"),
                       ("Không dấu", "dense|không dấu – thô", "v3|không dấu – khôi phục"),
                       ("Teencode", "dense|teencode – thô", "v3|teencode – chuẩn hóa")]:
        rows.append(f"| {name} | " + " | ".join(f"{pct(r[b][k])} → **{pct(r[v][k])}**"
                                                 for k in ("R@1", "R@5", "R@10", "MRR@10")) + " |")
    rel = "\n".join(f"| {x['level']} | {x['n']:.0f} | {pct(x['mean_conf'])} | **{pct(x['actual_acc'])}** |"
                    for x in norm_reliability(m["reliability"]))
    cols = ["Cao", "Trung bình", "Thấp", "Từ chối"]
    dec = "\n".join(f"| {k} | " + " | ".join(pct(v.get(c, 0)) for c in cols) + " |" for k, v in m["decisions"].items())
    s = m["safety"]
    saf = "\n".join(f"| {k} | {pct(s['baseline BTC'].get(k))} | **{pct(s['ViCare-Safe'].get(k))}** |"
                    for k in ("cấp cứu – có dấu", "cấp cứu – không dấu", "báo động giả"))
    saf += f"\n| chặn hỏi liều / kê đơn | – | **{pct(s['ViCare-Safe'].get('chặn hỏi liều/kê đơn'))}** |"
    rest, cal = m["restoration"], m["calibration"]
    return f"""{START}
Cùng kho tài liệu (9.895 tài liệu, 11.227 chunk) và **cùng 100 câu dev của notebook BTC**. Baseline = notebook gốc
(tìm kiếm vector e5). Cấu hình giống demo CPU: reranker chấm {m['config']['N_CAND']} ứng viên. Biến thể *không dấu* và
*teencode* được tạo tự động từ câu dev. Khoảng tin cậy 95% (bootstrap) và kiểm định McNemar: xem notebook, mục 5.

**Tìm đúng tài liệu (baseline → ViCare-Safe v3)**

| Cách gõ | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|
{chr(10).join(rows)}

**Khôi phục dấu:** {rest['syllable_acc'] * 100:.1f}% âm tiết đúng, {pct(rest['sentence_exact'])} câu đúng hoàn toàn trên 100 câu dev
(không dùng câu dev để học).

**Độ tin cậy có nói thật không?** (gộp mọi biến thể dev, các câu không bị từ chối; ECE = {cal['ECE']:.3f}, AUROC = {cal['AUROC']:.2f})

| Mức ViCare tự báo | Số câu | Độ tin cậy dự đoán TB | Tỷ lệ tìm đúng thực tế |
|---|---|---|---|
{rel}

**ViCare quyết định thế nào?** (cổng ngoài phạm vi τ = {m['tau']:.2f}, chọn trên tập tune để từ chối ≥ 95% câu ngoài phạm vi)

| Bộ câu hỏi | Cao | Trung bình | Thấp | Từ chối |
|---|---|---|---|---|
{dec}

**An toàn** (bộ câu tự viết – kết quả lạc quan)

| Kiểm tra | Baseline | ViCare-Safe |
|---|---|---|
{saf}
{END}"""


def main():
    m = json.loads((ROOT / "reports/metrics.json").read_text())
    readme = (ROOT / "README.md").read_text()
    block = render(m)
    if START in readme:
        readme = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, readme, flags=re.S)
    else:
        readme = readme.replace("<!-- RESULTS -->", block)
    (ROOT / "README.md").write_text(readme)
    print("✅ README cập nhật kết quả")


if __name__ == "__main__":
    main()
