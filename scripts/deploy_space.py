"""Đóng gói và triển khai demo lên Hugging Face Space.

Cần token Hugging Face quyền Write trong biến môi trường HF_TOKEN.
Tùy chọn: HF_SPACE_ID (mặc định "<tên-tài-khoản>/ViCare-Safe").
Chạy tay:  HF_TOKEN=hf_xxx python scripts/deploy_space.py   (hoặc tự động qua GitHub Actions)
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["app.py", "vicare_safe", "artifacts/e5_small_chunks.f16.npy", "artifacts/chunk_ids.json",
         "artifacts/confidence.json", "reports/metrics.json", "LICENSE"]
GRADIO_VERSION = "6.32.0"

SPACE_README = f"""---
title: ViCare-Safe
emoji: 🩺
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: {GRADIO_VERSION}
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Tra cứu sức khỏe tiếng Việt có độ tin cậy và bằng chứng
---

# ViCare-Safe

Trợ lý tra cứu sức khỏe tiếng Việt kiểu **hỗ trợ quyết định**: hiểu câu **không dấu** và **teencode**, trích **nguyên văn**
lời bác sĩ kèm **độ tin cậy**, **bằng chứng** và **nguồn**. Không chẩn đoán, không kê đơn. Cấp cứu: gọi **115**.

Mã nguồn & đánh giá: https://github.com/chinguyenkhac/ViCare
Dữ liệu: ViHealthQA (Nguyen et al., KSEM 2022) – chỉ dùng cho mục đích nghiên cứu.
"""

SPACE_REQUIREMENTS = """numpy>=1.26
scipy>=1.11
pandas>=2.0
datasets>=3,<5
sentence-transformers>=4,<6
torch>=2.2
"""


def main() -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()["name"]
    space_id = os.environ.get("HF_SPACE_ID") or f"{user}/ViCare-Safe"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for rel in FILES:
            src, dst = ROOT / rel, tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)
        (tmp / "README.md").write_text(SPACE_README, encoding="utf-8")
        (tmp / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
        api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(tmp), repo_id=space_id, repo_type="space",
                          commit_message="Deploy ViCare-Safe from GitHub")
    url = f"https://huggingface.co/spaces/{space_id}"
    docs = ROOT / "docs"
    if docs.exists():   # trang GitHub Pages đọc file này để bật nút "Mở demo trực tiếp"
        (docs / "space.json").write_text(json.dumps({"url": url}))
    print(f"✅ Space: {url}")


if __name__ == "__main__":
    main()
