"""
Task 5 — Semantic Search Module.

Dense retrieval trên ChromaDB collection đã index ở Task 4, dùng cùng
Cohere embedding model (`embed-multilingual-v3.0`) để embed query
(input_type="search_query" — khác với "search_document" lúc index, theo
khuyến nghị của Cohere để tối ưu match query↔document).
"""

import chromadb

from .task4_chunking_indexing import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    _cohere_client,
)

_collection = None


def _get_collection():
    """Lazy-load ChromaDB collection (tránh mở connection khi chỉ import module)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (cosine).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (1 - distance)
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    collection = _get_collection()

    query_embedding = _cohere_client.embed(
        texts=[query],
        model=EMBEDDING_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    ).embeddings.float_[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "content": doc,
            "score": 1 - distance,  # cosine distance → similarity
            "metadata": metadata,
        }
        for doc, distance, metadata in zip(
            results["documents"][0], results["distances"][0], results["metadatas"][0]
        )
    ]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
