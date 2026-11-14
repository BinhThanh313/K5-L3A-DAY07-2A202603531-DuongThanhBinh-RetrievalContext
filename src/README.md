# Code & Hướng dẫn chạy Day 7 — Agentic RAG

## 1. Cấu trúc mã nguồn (`src/`)
- `src/retrievers.py`: Triển khai các phương pháp retrieval (đều hỗ trợ metadata filter):
  - `BM25Retriever`: Lexical retrieval (Okapi BM25, chạy offline).
  - `OpenAIDenseRetriever`: Dense retrieval dùng `text-embedding-3-small` (1536 chiều, đồng bộ Day 6), cache vector tại `starter/data/converted/embeddings_cache.npz`.
  - `HybridRRFRetriever`: Hybrid kết hợp Dense + BM25 bằng Reciprocal Rank Fusion (`k=60`).
  - `RerankRetriever`: Lấy top-10 candidate từ Hybrid RRF rồi dùng `gpt-4o-mini` xếp hạng lại.
  - `QueryTransformRetriever`: Dùng `gpt-4o-mini` viết lại/mở rộng truy vấn rồi gộp kết quả bằng RRF.
- `src/agent.py`: `OpenAIController` (`gpt-4o-mini`, JSON mode) tương thích với `run_agent` trong `starter/agentic.py`.
- `src/benchmark_retrieval.py`: Benchmark so sánh các phương pháp retrieval trên `dev` / `test`.
- `src/run_agent_eval.py`: Chạy Agentic RAG trên `starter/data/converted/cases.json`, xuất `traces.json` và `reviews_draft.json`.

## 2. Cài đặt & Cấu hình kết nối
1. Cài thư viện (Python 3.11+):
   ```bash
   pip install openai numpy python-dotenv pypdf
   ```
2. Tạo file `.env` từ `.env.example` và điền OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-...
   OPENAI_EMBED_MODEL=text-embedding-3-small
   OPENAI_CHAT_MODEL=gpt-4o-mini
   ```

## 3. Dữ liệu & Tái tạo
- Dữ liệu nguồn từ Day 6: Báo cáo *World Bank Taking Stock March 2025 (Vietnam)* (giấy phép mở **CC BY 3.0 IGO**, 49 trang nội dung hợp lệ, 365 chunks chiến lược `by_paragraph`).
- Bộ dữ liệu đã chuẩn hóa tại `starter/data/converted/`:
  - `chunks.json`: 365 chunks kèm metadata (`section`, `sub_section`, `printed_page`, `page`, `source_id`).
  - `questions.json`: 20 câu có đáp án & gold evidence (`15 dev`, `5 test`).
  - `cases.json`: 24 câu cho Agent (`17 dev` = 15 có đáp án + 2 thiếu bằng chứng; `7 test` = 5 có đáp án + 2 thiếu bằng chứng).

## 4. Lệnh chạy kiểm chứng
```bash
# 1. Chạy unit test
python -m unittest discover -s tests -v

# 2. Chạy benchmark retrieval (tất cả phương pháp)
python -m src.benchmark_retrieval --split dev --budget 800
python -m src.benchmark_retrieval --split test --budget 3200

# 3. Chạy Agent cấu hình đã chọn (rerank, budget=3200, top_k=5)
python -m src.run_agent_eval --retriever rerank --split dev --budget 3200 --top-k 5 --max-searches 4 --output starter/results/agent_rerank_v2_dev
python -m src.run_agent_eval --retriever rerank --split test --budget 3200 --top-k 5 --max-searches 4 --output starter/results/agent_rerank_v2_test

# 4. Tổng hợp metrics từ nhãn audit (reviews.json)
python starter/agent_metrics.py --cases starter/data/converted/cases.json --traces starter/results/agent_rerank_v2_test/traces.json --reviews starter/results/agent_rerank_v2_test/reviews.json --split test --output starter/results/agent_rerank_v2_test/summary.json
```
