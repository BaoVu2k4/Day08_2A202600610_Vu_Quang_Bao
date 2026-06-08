"""Local RAG evaluation pipeline for the group project.

The assignment mentions DeepEval/RAGAS/TruLens. Those frameworks require
additional model-provider setup, so this script implements deterministic local
metrics with the same four evaluation axes:

- Faithfulness
- Answer Relevance
- Context Recall
- Context Precision

Run:
    python group_project/evaluation/eval_pipeline.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.task9_retrieval_pipeline import retrieve

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    "và", "là", "của", "có", "cho", "về", "theo", "đến", "trong", "một",
    "những", "các", "nào", "gì", "ở", "bị", "với", "được", "nêu", "biết",
    "quy", "định", "bài", "luật", "điều",
}


@dataclass
class EvalCaseResult:
    question: str
    expected_answer: str
    expected_context: str
    actual_output: str
    contexts: list[str]
    sources: list[str]
    faithfulness: float
    answer_relevance: float
    context_recall: float
    context_precision: float
    average: float
    error: str = ""


def load_golden_dataset() -> list[dict[str, Any]]:
    """Load golden dataset from JSON."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _overlap_score(reference: str, candidate: str) -> float:
    ref_tokens = _token_set(reference)
    if not ref_tokens:
        return 0.0
    candidate_tokens = _token_set(candidate)
    return len(ref_tokens & candidate_tokens) / len(ref_tokens)


def _f1_overlap(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    precision = len(a_tokens & b_tokens) / len(b_tokens)
    recall = len(a_tokens & b_tokens) / len(a_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _source_name(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {}) or {}
    return str(metadata.get("source", "unknown"))


def _extractive_answer(chunks: list[dict[str, Any]], max_chars: int = 1300) -> str:
    """Build a deterministic answer from retrieved context for eval."""
    if not chunks:
        return ""

    parts = []
    used = 0
    for chunk in chunks[:3]:
        source = _source_name(chunk)
        text = " ".join(chunk.get("content", "").split())
        remaining = max_chars - used
        if remaining <= 0:
            break
        snippet = text[:remaining]
        parts.append(f"{snippet} [{source}]")
        used += len(snippet)
    return "\n\n".join(parts)


def _score_case(item: dict[str, Any], chunks: list[dict[str, Any]]) -> EvalCaseResult:
    contexts = [chunk.get("content", "") for chunk in chunks]
    context_text = "\n".join(contexts)
    actual_output = _extractive_answer(chunks)
    expected_answer = item["expected_answer"]
    expected_context = item["expected_context"]

    # Faithfulness: how much of the generated extractive answer is supported by
    # retrieved contexts. It should be high because the answer is context-derived.
    faithfulness = _overlap_score(actual_output, context_text)

    # Answer relevance: expected answer and question terms should appear in the
    # actual output. We combine both so short legal answers are not over-penalized.
    answer_relevance = 0.7 * _overlap_score(expected_answer, actual_output)
    answer_relevance += 0.3 * _overlap_score(item["question"], actual_output)

    # Context recall: retrieved context should include expected answer evidence
    # and expected source markers.
    context_recall = 0.75 * _overlap_score(expected_answer, context_text)
    context_recall += 0.25 * _overlap_score(expected_context, context_text)

    # Context precision: each retrieved chunk should be useful for the question
    # or expected answer.
    if contexts:
        per_context = [
            max(
                _f1_overlap(item["question"], ctx),
                _f1_overlap(expected_answer, ctx),
                _f1_overlap(expected_context, ctx),
            )
            for ctx in contexts
        ]
        context_precision = statistics.mean(per_context)
    else:
        context_precision = 0.0

    scores = [faithfulness, answer_relevance, context_recall, context_precision]
    return EvalCaseResult(
        question=item["question"],
        expected_answer=expected_answer,
        expected_context=expected_context,
        actual_output=actual_output,
        contexts=contexts,
        sources=[_source_name(chunk) for chunk in chunks],
        faithfulness=round(faithfulness, 3),
        answer_relevance=round(answer_relevance, 3),
        context_recall=round(context_recall, 3),
        context_precision=round(context_precision, 3),
        average=round(statistics.mean(scores), 3),
    )


def evaluate_config(
    golden_dataset: list[dict[str, Any]],
    *,
    use_reranking: bool,
    top_k: int = 5,
) -> list[EvalCaseResult]:
    """Run retrieval and local metrics for one config."""
    results: list[EvalCaseResult] = []
    for i, item in enumerate(golden_dataset, 1):
        print(f"[{i}/{len(golden_dataset)}] {item['question'][:70]}...")
        try:
            chunks = retrieve(
                item["question"],
                top_k=top_k,
                score_threshold=0.0,
                use_reranking=use_reranking,
            )
            results.append(_score_case(item, chunks))
        except Exception as exc:
            results.append(
                EvalCaseResult(
                    question=item["question"],
                    expected_answer=item["expected_answer"],
                    expected_context=item["expected_context"],
                    actual_output="",
                    contexts=[],
                    sources=[],
                    faithfulness=0.0,
                    answer_relevance=0.0,
                    context_recall=0.0,
                    context_precision=0.0,
                    average=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def _aggregate(results: list[EvalCaseResult]) -> dict[str, float]:
    if not results:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "average": 0.0,
        }
    return {
        "faithfulness": round(statistics.mean(r.faithfulness for r in results), 3),
        "answer_relevance": round(statistics.mean(r.answer_relevance for r in results), 3),
        "context_recall": round(statistics.mean(r.context_recall for r in results), 3),
        "context_precision": round(statistics.mean(r.context_precision for r in results), 3),
        "average": round(statistics.mean(r.average for r in results), 3),
    }


def compare_configs(golden_dataset: list[dict[str, Any]]) -> dict[str, list[EvalCaseResult]]:
    """Compare at least two RAG configs."""
    configs = {
        "hybrid_rerank": {"use_reranking": True},
        "hybrid_no_rerank": {"use_reranking": False},
    }
    comparison = {}
    for name, params in configs.items():
        print(f"\n=== Evaluating {name} ===")
        comparison[name] = evaluate_config(golden_dataset, **params)
    return comparison


def _metric_row(label: str, key: str, agg_a: dict[str, float], agg_b: dict[str, float]) -> str:
    delta = agg_a[key] - agg_b[key]
    return f"| {label} | {agg_a[key]:.3f} | {agg_b[key]:.3f} | {delta:+.3f} |"


def export_results(comparison: dict[str, list[EvalCaseResult]]) -> None:
    """Export evaluation results to results.md."""
    primary = comparison["hybrid_rerank"]
    baseline = comparison["hybrid_no_rerank"]
    agg_primary = _aggregate(primary)
    agg_baseline = _aggregate(baseline)

    worst = sorted(primary, key=lambda r: r.average)[:3]
    error_count = sum(1 for results in comparison.values() for r in results if r.error)

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "Local deterministic evaluator with 4 RAG metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision.",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Delta |",
        "|--------|-----------------------------|------------------------------|-------|",
        _metric_row("Faithfulness", "faithfulness", agg_primary, agg_baseline),
        _metric_row("Answer Relevance", "answer_relevance", agg_primary, agg_baseline),
        _metric_row("Context Recall", "context_recall", agg_primary, agg_baseline),
        _metric_row("Context Precision", "context_precision", agg_primary, agg_baseline),
        _metric_row("Average", "average", agg_primary, agg_baseline),
        "",
        "## A/B Comparison Analysis",
        "",
        "**Config A:** Semantic search + BM25 lexical search + RRF merge + Cohere reranking.",
        "",
        "**Config B:** Semantic search + BM25 lexical search + RRF merge, no cross-encoder reranking.",
        "",
        "**Kết luận:** Config A được chọn cho demo chính vì reranker chấm lại query-document theo cặp, thường giúp đưa context sát câu hỏi lên đầu. Config B là baseline nhanh hơn và ít tốn API hơn, hữu ích khi cần giảm latency.",
        "",
        "## Worst Performers (Bottom 3, Config A)",
        "",
        "| # | Question | Faithfulness | Relevance | Recall | Precision | Top Sources | Root Cause |",
        "|---|----------|--------------|-----------|--------|-----------|-------------|------------|",
    ]

    for i, item in enumerate(worst, 1):
        sources = ", ".join(item.sources[:3]) if item.sources else "none"
        root_cause = "Retriever chưa lấy đủ evidence" if item.context_recall < 0.5 else "Context đúng nhưng còn nhiễu"
        if item.error:
            root_cause = item.error
        lines.append(
            f"| {i} | {item.question} | {item.faithfulness:.3f} | "
            f"{item.answer_relevance:.3f} | {item.context_recall:.3f} | "
            f"{item.context_precision:.3f} | {sources} | {root_cause} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "### Cải tiến 1",
            "**Action:** Bổ sung thêm văn bản pháp luật text-based, nhất là nghị định/danh mục chất ma túy và văn bản hướng dẫn cai nghiện.",
            "**Expected impact:** Tăng Context Recall cho câu hỏi pháp luật có điều khoản cụ thể.",
            "",
            "### Cải tiến 2",
            "**Action:** Thêm metadata chi tiết hơn khi chunk, gồm điều/chương, tên báo, năm xuất bản và URL.",
            "**Expected impact:** Citation rõ hơn và giảm nhầm nguồn khi generation.",
            "",
            "### Cải tiến 3",
            "**Action:** Tách legal/news thành hai retriever có routing theo intent trước khi RRF.",
            "**Expected impact:** Giảm nhiễu giữa câu hỏi điều luật và câu hỏi tin tức nghệ sĩ.",
            "",
            "## Run Notes",
            "",
            f"- Total test cases: {len(primary)}",
            f"- Errors captured: {error_count}",
            "- PageIndex fallback is disabled during evaluation by using score_threshold=0.0, because the available account may hit retrieval quota limits.",
            "",
        ]
    )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report: {RESULTS_PATH}")


def main() -> None:
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    comparison = compare_configs(golden_dataset)
    export_results(comparison)


if __name__ == "__main__":
    main()
