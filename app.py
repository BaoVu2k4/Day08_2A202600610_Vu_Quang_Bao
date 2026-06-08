"""Streamlit chatbot for the group RAG demo.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve

load_dotenv()


st.set_page_config(
    page_title="Drug Law RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


def _source_label(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source", "unknown")
    doc_type = metadata.get("type", "unknown")
    score = chunk.get("score", 0.0)
    return f"{source} | {doc_type} | score={score:.3f}"


def _build_followup_query(question: str) -> str:
    """Add short conversation context so follow-up questions are answerable."""
    history = st.session_state.messages[-6:]
    if not history:
        return question

    turns = []
    for message in history:
        role = "Người dùng" if message["role"] == "user" else "Trợ lý"
        turns.append(f"{role}: {message['content']}")
    return "\n".join(turns + [f"Người dùng: {question}"])


def _answer_question(question: str, top_k: int, generation_enabled: bool) -> dict[str, Any]:
    expanded_query = _build_followup_query(question)

    if generation_enabled:
        result = generate_with_citation(expanded_query, top_k=top_k)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "retrieval_source": result.get("retrieval_source", "hybrid"),
        }

    chunks = retrieve(expanded_query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Không tìm thấy ngữ cảnh phù hợp trong kho dữ liệu hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    preview = "\n\n".join(
        f"- [{_source_label(chunk)}] {chunk['content'][:450].strip()}..."
        for chunk in chunks
    )
    return {
        "answer": "Chế độ retrieval-only đang bật. Các đoạn liên quan nhất:\n\n" + preview,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


def main() -> None:
    _init_state()

    st.title("Drug Law RAG Chatbot")

    with st.sidebar:
        st.header("Cấu hình")
        top_k = st.slider("Số nguồn", min_value=3, max_value=8, value=5)
        has_groq_key = bool(os.getenv("GROQ_API_KEY"))
        generation_enabled = st.toggle(
            "Sinh câu trả lời bằng LLM",
            value=has_groq_key,
            disabled=not has_groq_key,
            help="Tắt để demo retrieval/source khi chưa có GROQ_API_KEY.",
        )
        if not has_groq_key:
            st.info("Thiếu GROQ_API_KEY, app chạy ở chế độ retrieval-only.")

        if st.button("Xóa hội thoại", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

        st.divider()
        st.caption("Nguồn gần nhất")
        for i, chunk in enumerate(st.session_state.last_sources, 1):
            st.write(f"{i}. {_source_label(chunk)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Hỏi về pháp luật ma túy hoặc các bài báo trong corpus")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            try:
                result = _answer_question(question, top_k=top_k, generation_enabled=generation_enabled)
                answer = result["answer"]
                st.markdown(answer)

                sources = result.get("sources", [])
                st.session_state.last_sources = sources
                if sources:
                    with st.expander("Nguồn đã sử dụng", expanded=True):
                        for i, chunk in enumerate(sources, 1):
                            st.markdown(f"**{i}. {_source_label(chunk)}**")
                            st.write(chunk["content"][:900])

            except Exception as exc:
                answer = f"Lỗi khi chạy pipeline: `{type(exc).__name__}: {exc}`"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
