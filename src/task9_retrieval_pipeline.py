"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search
    2. Merge kết quả bằng RRF (Reciprocal Rank Fusion — công bằng vì dựa
       trên rank thay vì so trực tiếp cosine similarity với BM25 score,
       2 thang điểm hoàn toàn khác nhau)
    3. Rerank bằng Cohere cross-encoder (chấm lại độ liên quan chính xác hơn)
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def _safe_pageindex_search(query: str, top_k: int) -> list[dict]:
    """
    Wrapper quanh pageindex_search — fallback không được phép làm sập pipeline
    (vd: thiếu PAGEINDEX_API_KEY, document chưa xử lý xong, lỗi mạng...).
    """
    try:
        return pageindex_search(query, top_k=top_k)
    except Exception as e:
        print(f"  ⚠ PageIndex fallback lỗi ({e}) — giữ nguyên kết quả hybrid")
        return []


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results (so trên
            rerank relevance_score — thang 0..1, Cohere rerank-multilingual-v3.0)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Song song chạy semantic + lexical (lấy dư top_k*2 để RRF có đủ
    # ứng viên trước khi rerank thu hẹp lại)
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    if not merged:
        return _safe_pageindex_search(query, top_k=top_k)

    # Step 3: Rerank
    if use_reranking:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Step 4: Check threshold → fallback
    if not final_results or final_results[0]["score"] < score_threshold:
        best = final_results[0]["score"] if final_results else 0.0
        print(f"  ⚠ Hybrid score ({best:.3f}) < threshold ({score_threshold}). Fallback → PageIndex")
        fallback = _safe_pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback
        # PageIndex cũng không có gì → trả lại hybrid results (tốt hơn rỗng)
        return final_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
