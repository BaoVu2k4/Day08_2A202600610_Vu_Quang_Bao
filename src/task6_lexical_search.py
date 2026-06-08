"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25 (rank-bm25). Corpus được load trực tiếp từ
data/standardized/ + chunk lại bằng cùng splitter ở Task 4, để khớp 1-1
với granularity của semantic search (so sánh công bằng khi merge ở Task 9).

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization) — mặc định BM25Okapi
"""

import re

from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import chunk_documents, load_documents

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản: lowercase + tách theo word boundary (giữ dấu tiếng Việt)."""
    return _TOKEN_RE.findall(text.lower())


_corpus = None
_bm25 = None


def _load_corpus() -> list[dict]:
    """Load + chunk toàn bộ documents từ data/standardized/ (lazy, cache 1 lần)."""
    global _corpus
    if _corpus is None:
        _corpus = chunk_documents(load_documents())
    return _corpus


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _get_index():
    """Lazy-build BM25 index (cache 1 lần cho cả module)."""
    global _bm25
    if _bm25 is None:
        _bm25 = build_bm25_index(_load_corpus())
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25 = _get_index()
    corpus = _load_corpus()

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked[:top_k]:
        if scores[idx] <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
