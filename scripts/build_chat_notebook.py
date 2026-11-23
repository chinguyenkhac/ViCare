"""Sinh notebooks/ViCare_Chatbot_Colab.ipynb – chatbot hỏi đáp chạy trực tiếp trên Google Colab."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/chinguyenkhac/ViCare"
COLAB = "https://colab.research.google.com/github/chinguyenkhac/ViCare/blob/main/notebooks/ViCare_Chatbot_Colab.ipynb"

INTRO = f"""# ViCare-Safe – Chatbot hỏi đáp (chạy trên Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

**Cách dùng (3 bước):**
1. Bấm **Runtime → Run all** (hoặc chạy lần lượt 2 ô bên dưới).
2. Đợi ô 2 in ra `Running on public URL: https://….gradio.live` (lần đầu khoảng 2–4 phút: tải mô hình và dữ liệu).
3. Gõ câu hỏi ngay trong khung chat bên dưới ô 2, hoặc mở link `gradio.live` ở tab mới.

Gõ **có dấu, không dấu hoặc teencode** đều được, ví dụ: `con toi 2 tuoi sot cao co giat phai lam sao`.
Hệ thống hiểu lại câu hỏi, báo **độ tin cậy** (Cao / Trung bình / Thấp), trích **nguyên văn** lời bác sĩ kèm nguồn,
cảnh báo cấp cứu (115), từ chối câu ngoài phạm vi và chặn câu hỏi xin liều thuốc.

> ⚠️ Sản phẩm minh họa học thuật – không chẩn đoán, không kê đơn, không thay thế nhân viên y tế.
> Chế độ trích xuất, chạy trên CPU, không dùng LLM sinh văn bản. Mỗi câu mất khoảng 10–60 giây.
> Link `gradio.live` chỉ sống khi phiên Colab còn chạy (tối đa 1 tuần).
> Mã nguồn: {REPO}"""

SETUP = """import os, subprocess, sys
from pathlib import Path

if not Path("vicare_safe").exists():
    if not Path("ViCare").exists():
        subprocess.run(["git", "clone", "--depth", "1", "%s"], check=True)
    os.chdir("ViCare")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("✅ Cài đặt xong, thư mục:", os.getcwd())""" % REPO

LAUNCH = """import gradio as gr
import app  # nạp kho tài liệu, mô hình tìm kiếm, reranker (lần đầu ~2–4 phút)

app.demo.queue(default_concurrency_limit=1).launch(
    share=True,      # tạo link công khai gradio.live
    inline=True,     # hiện luôn khung chat trong notebook
    height=800,
    theme=gr.themes.Soft(),
)"""


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(INTRO),
                nbf.v4.new_markdown_cell("## 1. Cài đặt"), nbf.v4.new_code_cell(SETUP),
                nbf.v4.new_markdown_cell("## 2. Bật chatbot"), nbf.v4.new_code_cell(LAUNCH)]
    nb.metadata = {"colab": {"name": "ViCare_Chatbot_Colab.ipynb", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    out = ROOT / "notebooks/ViCare_Chatbot_Colab.ipynb"
    nbf.write(nb, out)
    print("✅ đã ghi", out)


if __name__ == "__main__":
    main()
