"""
Task 8 — PageIndex Vectorless RAG.

PageIndex làm RAG dựa trên cấu trúc tài liệu (tree từ mục lục/heading) thay
vì vector embedding — phù hợp nhất với văn bản có cấu trúc rõ ràng như văn
bản pháp luật (Điều/Khoản/Chương). Vì vậy ta upload trực tiếp các file PDF
gốc trong data/landing/legal/ (PageIndex SDK chỉ nhận PDF, tự OCR + dựng
tree), thay vì các .md đã convert ở Task 3.

LƯU Ý: Cần PAGEINDEX_API_KEY trong .env (đăng ký tại https://pageindex.ai/).
Chưa có key nên phần upload/query CHƯA chạy thử được trên máy này — code viết
theo đúng signature của SDK `pageindex` v0.2.8 (PageIndexClient), khi có key
chỉ cần `python -m src.task8_pageindex_vectorless` để upload + test query.

Cài đặt:
    pip install pageindex
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "vectorstore" / "pageindex_docs.json"

POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 300

_client = None


def _get_client() -> PageIndexClient:
    global _client
    if _client is None:
        if not PAGEINDEX_API_KEY:
            raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env — đăng ký tại https://pageindex.ai/")
        _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _client


def _load_registry() -> dict:
    """doc_registry: {filename: doc_id} — cache để khỏi upload lại mỗi lần chạy."""
    if DOC_REGISTRY_PATH.exists():
        return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _save_registry(registry: dict):
    DOC_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents() -> dict:
    """
    Upload toàn bộ PDF pháp luật trong data/landing/legal/ lên PageIndex.

    Returns:
        dict {filename: doc_id} — registry các document đã upload.
    """
    client = _get_client()
    registry = _load_registry()

    for pdf_file in LANDING_LEGAL_DIR.glob("*.pdf"):
        if pdf_file.name in registry:
            print(f"  ↷ Đã upload trước đó, bỏ qua: {pdf_file.name} (doc_id={registry[pdf_file.name]})")
            continue

        print(f"  Uploading: {pdf_file.name}...")
        result = client.submit_document(file_path=str(pdf_file))
        doc_id = result["doc_id"]
        registry[pdf_file.name] = doc_id
        _save_registry(registry)
        print(f"  ✓ Uploaded: {pdf_file.name} → doc_id={doc_id}")

    print("\n  Chờ PageIndex xử lý OCR + dựng tree (retrieval_ready)...")
    for filename, doc_id in registry.items():
        deadline = time.time() + POLL_TIMEOUT_SECONDS
        while time.time() < deadline:
            if client.is_retrieval_ready(doc_id):
                print(f"  ✓ Sẵn sàng retrieval: {filename}")
                break
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            print(f"  ⚠ Timeout chờ xử lý: {filename} (doc_id={doc_id})")

    return registry


def _wait_for_retrieval(client: PageIndexClient, retrieval_id: str) -> dict:
    """Poll get_retrieval cho tới khi có kết quả hoặc timeout."""
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        if result.get("status") in ("completed", "ready", "done") or result.get("results"):
            return result
        time.sleep(POLL_INTERVAL_SECONDS)
    return {}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Truy vấn song song trên từng document đã upload (PageIndex retrieval scoped
    theo doc_id), gộp kết quả từ tất cả documents rồi sort theo relevance.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    client = _get_client()
    registry = _load_registry()
    if not registry:
        registry = upload_documents()

    all_results = []
    for filename, doc_id in registry.items():
        submitted = client.submit_query(doc_id=doc_id, query=query)
        retrieval = _wait_for_retrieval(client, submitted["retrieval_id"])

        for node in retrieval.get("results", []):
            all_results.append({
                "content": node.get("content") or node.get("text", ""),
                "score": float(node.get("relevance_score", node.get("score", 0.0))),
                "metadata": {"source": filename, "node_id": node.get("node_id"), "type": "legal"},
                "source": "pageindex",
            })

    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
