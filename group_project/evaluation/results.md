# RAG Evaluation Results

## Framework sử dụng

Local deterministic evaluator with 4 RAG metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision.

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid no rerank) | Delta |
|--------|-----------------------------|------------------------------|-------|
| Faithfulness | 0.961 | 0.959 | +0.002 |
| Answer Relevance | 0.815 | 0.814 | +0.001 |
| Context Recall | 0.838 | 0.850 | -0.012 |
| Context Precision | 0.256 | 0.254 | +0.002 |
| Average | 0.718 | 0.719 | -0.001 |

## A/B Comparison Analysis

**Config A:** Semantic search + BM25 lexical search + RRF merge + Cohere reranking.

**Config B:** Semantic search + BM25 lexical search + RRF merge, no cross-encoder reranking.

**Kết luận:** Config A được chọn cho demo chính vì reranker chấm lại query-document theo cặp, thường giúp đưa context sát câu hỏi lên đầu. Config B là baseline nhanh hơn và ít tốn API hơn, hữu ích khi cần giảm latency.

## Worst Performers (Bottom 3, Config A)

| # | Question | Faithfulness | Relevance | Recall | Precision | Top Sources | Root Cause |
|---|----------|--------------|-----------|--------|-----------|-------------|------------|
| 1 | Điều 250 Bộ luật Hình sự liên quan đến tội gì? | 0.949 | 0.370 | 0.417 | 0.167 | bo-luat-hinh-su-2015.md, bo-luat-hinh-su-2015.md, bo-luat-hinh-su-2015.md | Retriever chưa lấy đủ evidence |
| 2 | Điều 248 Bộ luật Hình sự quy định về hành vi nào? | 0.938 | 0.456 | 0.750 | 0.150 | bo-luat-hinh-su-2015.md, bo-luat-hinh-su-2015.md, bo-luat-hinh-su-2015.md | Context đúng nhưng còn nhiễu |
| 3 | Bài Tuổi Trẻ về Bình Gold cho biết rapper này bị bắt vì lý do gì? | 0.962 | 0.700 | 0.714 | 0.283 | article_05.md, article_06.md, article_07.md | Context đúng nhưng còn nhiễu |

## Recommendations

### Cải tiến 1
**Action:** Bổ sung thêm văn bản pháp luật text-based, nhất là nghị định/danh mục chất ma túy và văn bản hướng dẫn cai nghiện.
**Expected impact:** Tăng Context Recall cho câu hỏi pháp luật có điều khoản cụ thể.

### Cải tiến 2
**Action:** Thêm metadata chi tiết hơn khi chunk, gồm điều/chương, tên báo, năm xuất bản và URL.
**Expected impact:** Citation rõ hơn và giảm nhầm nguồn khi generation.

### Cải tiến 3
**Action:** Tách legal/news thành hai retriever có routing theo intent trước khi RRF.
**Expected impact:** Giảm nhiễu giữa câu hỏi điều luật và câu hỏi tin tức nghệ sĩ.

## Run Notes

- Total test cases: 15
- Errors captured: 0
- PageIndex fallback is disabled during evaluation by using score_threshold=0.0, because the available account may hit retrieval quota limits.
