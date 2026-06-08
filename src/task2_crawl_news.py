"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Lưu output vào data/landing/news/
    3. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Ghi chú lựa chọn công cụ:
    Crawl4AI yêu cầu cài Playwright + tải browser binary (~vài trăm MB), khá nặng
    cho máy cá nhân. Thay vào đó dùng Jina AI Reader (https://r.jina.ai/<url>) —
    một dịch vụ "URL-to-LLM-friendly-markdown" miễn phí, trả về markdown đã làm
    sạch HTML, không cần chạy browser. Header X-Target-Selector cho phép chỉ định
    CSS selector để lấy đúng phần thân bài viết (loại bỏ menu/sidebar/quảng cáo).
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_READER_URL = "https://r.jina.ai/"
HEADERS_BASE = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# CSS selector cho phần thân bài viết — riêng theo từng trang báo, giúp Jina Reader
# bỏ qua menu/sidebar/footer và chỉ trích nội dung chính.
ARTICLE_SELECTORS = {
    "vnexpress.net": "article.fck_detail",
    "vietnamnet.vn": "div.maincontent, div.ArticleContent, article",
    "vov.vn": "div.article-content, div#article-body, article",
    "tuoitre.vn": "div#main-detail-body, div.detail-content, article",
    "thanhnien.vn": "div.detail-content, div#abody, article",
}


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý (2018-2026), từ các báo uy tín.
# Đa dạng nhiều nghệ sĩ/vụ việc khác nhau (không chỉ tập trung vào 1 người):
#   - Miu Lê (bị bắt quả tang sử dụng ma tuý, 2025)
#   - Châu Việt Cường (ngáo đá, gây ra cái chết của 1 cô gái, 2018)
#   - Chi Dân & An Tây & Trúc Phương (đường dây ma tuý "VN10", 2024)
#   - Bình Gold (rapper, dương tính ma tuý + cướp tài sản, 2025)
#   - 2 bài tổng hợp nhiều nghệ sĩ Việt dính ma tuý qua các thời kỳ
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-miu-le-bi-bat-qua-tang-dung-ma-tuy-o-bai-bien-5072657.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-hau-toa-vi-nhet-toi-hai-chet-co-gai-20-tuoi-3890738.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
    "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
    "https://tuoitre.vn/rapper-binh-gold-bi-bat-vi-cuop-tai-san-duong-tinh-voi-ma-tuy-20250726185902989.htm",
    "https://vietnamnet.vn/rapper-binh-gold-bi-bat-vi-cuop-taxi-duong-tinh-voi-ma-tuy-2426027.html",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
]


def _selector_for(url: str) -> str | None:
    """Tìm CSS selector phù hợp dựa trên domain của URL."""
    for domain, selector in ARTICLE_SELECTORS.items():
        if domain in url:
            return selector
    return None


def _parse_jina_response(raw: str) -> dict:
    """
    Tách response của Jina Reader (dạng text) thành title + markdown content.

    Format trả về:
        Title: ...
        URL Source: ...
        Published Time: ...
        Markdown Content:
        <nội dung markdown>
    """
    title = "Unknown"
    body = raw

    marker = "Markdown Content:"
    if marker in raw:
        header, body = raw.split(marker, 1)
        body = body.strip()
        for line in header.splitlines():
            if line.startswith("Title:"):
                title = line.removeprefix("Title:").strip()
                break

    return {"title": title, "content_markdown": body}


def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo qua Jina AI Reader và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    headers = dict(HEADERS_BASE)
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"

    selector = _selector_for(url)
    if selector:
        headers["X-Target-Selector"] = selector

    response = requests.get(JINA_READER_URL + url, headers=headers, timeout=120)

    # Một số trang dùng template khác không khớp selector → Jina trả 422.
    # Fallback: thử lại không kèm selector (lấy nguyên trang rồi parse).
    if response.status_code == 422 and selector:
        headers.pop("X-Target-Selector", None)
        response = requests.get(JINA_READER_URL + url, headers=headers, timeout=120)

    response.raise_for_status()

    parsed = _parse_jina_response(response.text)
    return {
        "url": url,
        "title": parsed["title"],
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": parsed["content_markdown"],
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = crawl_article(url)
        except Exception as exc:
            print(f"  x Lỗi khi crawl {url}: {exc}")
            continue

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  - Saved: {filepath.name} | title: {article['title'][:60]}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        crawl_all()
