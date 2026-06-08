"""
Task 4 — Chunking & Indexing vào Vector Store.

Pipeline: load markdown từ data/standardized/ → chunk → embed (Cohere) → index (ChromaDB).

Lựa chọn & lý do:
    - Chunking: RecursiveCharacterTextSplitter, chunk_size=500, overlap=50.
      500 ký tự ~ 1 đoạn văn bản luật/tin tức tiếng Việt (đủ ngữ cảnh cho 1 ý,
      không quá dài gây loãng embedding). Overlap=50 (10%) giữ liên kết ngữ
      nghĩa giữa các chunk liền kề, tránh cắt đứt câu ở ranh giới.
    - Embedding: Cohere `embed-multilingual-v3.0` (1024 dim) qua API — đã có
      COHERE_API_KEY sẵn, multilingual, chất lượng tốt cho tiếng Việt, không
      cần tải model nặng (~2GB) về máy không có GPU.
    - Vector store: ChromaDB (PersistentClient, lưu file local trong
      data/vectorstore/) — embedded, không cần Docker/server, phù hợp máy hiện tại
      (Weaviate cần Docker đang không chạy).

Cài đặt:
    pip install langchain-text-splitters cohere chromadb
"""

import os
import time
from pathlib import Path

import chromadb
import cohere
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTORSTORE_DIR = Path(__file__).parent.parent / "data" / "vectorstore"

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 500        # ~1 đoạn văn bản luật/tin tức tiếng Việt — đủ ngữ cảnh cho 1 ý
CHUNK_OVERLAP = 50      # 10% overlap — giữ liên kết ngữ nghĩa, tránh cắt đứt câu
CHUNKING_METHOD = "recursive"  # RecursiveCharacterTextSplitter — an toàn, phổ biến

EMBEDDING_MODEL = "embed-multilingual-v3.0"  # Cohere — multilingual, tốt cho tiếng Việt
EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 90   # Cohere embed API giới hạn ~96 texts/request

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "drug_law_docs"

_cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.relative_to(STANDARDIZED_DIR)) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def _embed_with_retry(batch: list[str], input_type: str, max_retries: int = 6):
    """Gọi Cohere embed với retry/backoff — trial key giới hạn ~100k tokens/phút."""
    for attempt in range(max_retries):
        try:
            return _cohere_client.embed(
                texts=batch,
                model=EMBEDDING_MODEL,
                input_type=input_type,
                embedding_types=["float"],
            )
        except cohere.errors.too_many_requests_error.TooManyRequestsError:
            wait = 20 * (attempt + 1)
            print(f"    ⚠ Rate limited, chờ {wait}s rồi thử lại (lần {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("Cohere embed thất bại sau nhiều lần retry (rate limit)")


def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Gọi Cohere embed API theo batch (giới hạn EMBED_BATCH_SIZE texts/request)."""
    embeddings = []
    n_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for batch_idx, i in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), 1):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        print(f"    Embedding batch {batch_idx}/{n_batches} ({len(batch)} chunks)...")
        response = _embed_with_retry(batch, input_type)
        embeddings.extend(response.embeddings.float_)
        time.sleep(2)  # spacing giữa các batch để tránh chạm rate limit token/phút
    return embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng Cohere (input_type="search_document").

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts, input_type="search_document")
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks (content + embedding + metadata) vào ChromaDB collection local."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Reset collection để index lại từ đầu cho idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}::{c['metadata']['chunk_index']}" for c in chunks]
    collection.add(
        ids=ids,
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print(f"✓ Indexed to {VECTOR_STORE} at {VECTORSTORE_DIR}")


if __name__ == "__main__":
    run_pipeline()
