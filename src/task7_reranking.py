"""
Task 7 — Reranking Module.

Phương pháp chính: Cross-encoder reranker qua Cohere Rerank API
(`rerank-multilingual-v3.0`) — đã có COHERE_API_KEY sẵn, multilingual,
chấm lại độ liên quan câu query↔document chính xác hơn nhiều so với
cosine similarity / BM25 đơn thuần (vì model thấy cả query và document
cùng lúc thay vì so 2 vector độc lập).

RRF (Reciprocal Rank Fusion) cũng được implement — dùng ở Task 9 để gộp
kết quả semantic search + lexical search trước khi đưa vào cross-encoder.
"""

import os
import time

import cohere
from dotenv import load_dotenv

load_dotenv()

RERANK_MODEL = "rerank-multilingual-v3.0"

_cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))


def _rerank_with_retry(query: str, documents: list[str], top_n: int, max_retries: int = 4):
    """Gọi Cohere rerank với retry/backoff — trial key có rate limit theo phút."""
    for attempt in range(max_retries):
        try:
            return _cohere_client.rerank(
                query=query,
                documents=documents,
                model=RERANK_MODEL,
                top_n=top_n,
            )
        except cohere.errors.too_many_requests_error.TooManyRequestsError:
            wait = 15 * (attempt + 1)
            print(f"    ⚠ Rerank rate limited, chờ {wait}s rồi thử lại (lần {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("Cohere rerank thất bại sau nhiều lần retry (rate limit)")


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Cohere cross-encoder reranker.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by relevance_score descending.
    """
    if not candidates:
        return []

    response = _rerank_with_retry(
        query=query,
        documents=[c["content"] for c in candidates],
        top_n=min(top_k, len(candidates)),
    )

    return [
        {**candidates[r.index], "score": r.relevance_score}
        for r in response.results
    ]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector — dùng cho MMR."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Mỗi document được chấm điểm dựa trên THỨ HẠNG (rank) của nó trong từng
    ranked list, không phải raw score — nhờ đó gộp được kết quả từ các
    ranker có thang điểm khác nhau (cosine similarity vs BM25 score) một
    cách công bằng. k=60 là constant làm mượt, lấy từ paper Cormack et al. 2009.

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = dict(content_map[content])
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise ValueError("method='mmr' cần query_embedding — gọi rerank_mmr() trực tiếp")
    elif method == "rrf":
        raise ValueError("method='rrf' cần nhiều ranked_lists — gọi rerank_rrf() trực tiếp")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
