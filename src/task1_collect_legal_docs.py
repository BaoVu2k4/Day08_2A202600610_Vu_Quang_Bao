"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Direct links lấy từ Cổng Thông tin điện tử Chính phủ (chinhphu.vn / datafiles.chinhphu.vn),
# nguồn chính thống — đã verify tải về thành công (HTTP 200, application/pdf).
LEGAL_DOCS = [
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf",
        "filename": "bo-luat-hinh-su-2015.pdf",
    },
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/08/57-cp.signed.pdf",
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.pdf",
    },
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str):
    """Tải 1 file PDF/DOCX từ direct link và lưu vào DATA_DIR."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"  ↷ Đã tồn tại, bỏ qua: {filepath.name}")
        return

    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    filepath.write_bytes(response.content)
    print(f"  ✓ Đã tải ({len(response.content):,} bytes): {filepath.name}")


def download_all():
    """Tải toàn bộ văn bản pháp luật trong LEGAL_DOCS."""
    setup_directory()
    for doc in LEGAL_DOCS:
        print(f"Downloading: {doc['filename']}")
        download_file(doc["url"], doc["filename"])


if __name__ == "__main__":
    download_all()
