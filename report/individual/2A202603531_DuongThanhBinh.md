# Báo cáo Day 7 — Agentic RAG (Retrieval & Context Engineering)

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tài liệu từ Day 6: Báo cáo *World Bank Taking Stock March 2025 (Vietnam)* (`TAKING STOCK March 2025_VNM_FINAL.pdf`), giấy phép mở **CC BY 3.0 IGO** (World Bank Open Knowledge Repository).
  - Số trang: 64 trang vật lý, lọc bỏ 15 trang phi nội dung (bìa, mục lục, trang trống), còn **49 trang nội dung hợp lệ**.
  - Chiến lược chunking từ Day 6 (`by_paragraph`, Strategy B): **365 chunks**, lưu tại `starter/data/converted/chunks.json` với đầy đủ `id`, `source_id`, `source`, `page`, `text`, và `metadata` (`section`, `sub_section`, `printed_page`, `year`).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Embedding model: OpenAI `text-embedding-3-small` (**1536 chiều**), tương thích 100% với Day 6.
  - Vector store: Lưu trữ vector chuẩn hóa L2 dạng local cache (`starter/data/converted/embeddings_cache.npz`) dùng `numpy` cosine similarity và hỗ trợ lọc metadata chính xác theo `section`.
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent LLM: OpenAI `gpt-4o-mini` (JSON mode, `temperature=0.0`) tích hợp qua `OpenAIController` trong `src/agent.py` và vòng lặp `run_agent` của `starter/agentic.py`.
  - Bộ câu hỏi chuẩn Day 7 (`starter/data/converted/cases.json`): **24 câu**, chia cố định trước khi chạy:
    - **Dev (17 câu):** 15 câu có đáp án (`q01`–`q15` từ Day 6) + 2 câu thiếu bằng chứng (`q21`, `q22`).
    - **Test (7 câu):** 5 câu có đáp án (`q16`–`q20` từ Day 6) + 2 câu thiếu bằng chứng (`q23`, `q24`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - 20 câu có đáp án giữ nguyên `gold_source_id`, `gold_page`, `evidence_quote` (và `alternative_gold`) từ Day 6; đã kiểm tra khớp 100% chuỗi bằng chứng với `chunks.json`.
  - 4 câu thiếu bằng chứng được thiết kế cùng chủ đề kinh tế/xe điện Việt Nam nhưng kiểm tra chắc chắn từ khóa không tồn tại trong toàn bộ 365 chunks (ví dụ: doanh số *VinFast* Q1/2025, CPI *quý IV* 2025, xuất khẩu *sầu riêng*, *trợ cấp trực tiếp* tiền mặt 2025).
  - Tiêu chí chọn cấu hình trên `dev`: Ưu tiên `Grounded answer mean` và `False abstention` thấp nhất, kế đến là `Hit@1` / `Context Evidence Rate` và `Citation validity`, cân bằng với độ trễ và chi phí.

---

## 2. Retrieval và thử nghiệm

### Bảng so sánh các phương pháp Retrieval & cấu hình

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (Lexical) | TF-IDF, `top_k=3`, không filter, `budget=800` | Baseline offline từ `starter/baseline.py` | dev | 10/15 (66.7%) | 12/15 (80.0%) | 10/15 (66.7%) | 1.33 ms / \$0.0 |
| Phương pháp 2 (Lexical) | BM25 (`k1=1.5, b=0.75`), `top_k=3`, có filter, `budget=800` | Chuẩn hóa độ dài đoạn văn + filter giúp tăng Hit@3 | dev | 10/15 (66.7%) | 15/15 (100.0%) | 10/15 (66.7%) | 1.51 ms / \$0.0 |
| Phương pháp 3 (Dense) | OpenAI `text-embedding-3-small`, `top_k=3`, có filter, `budget=800` | Ngữ nghĩa vector giúp hiểu câu hỏi diễn đạt lại | dev | 10/15 (66.7%) | 14/15 (93.3%) | 10/15 (66.7%) | 440.7 ms / \$0.000023 |
| Cải thiện 1 (Hybrid) | Hybrid RRF (Dense + BM25, `k=60`), `top_k=3`, có filter, `budget=800` | Kết hợp từ khóa chính xác (BM25) và ngữ nghĩa (Dense) | dev | 11/15 (73.3%) | 15/15 (100.0%) | 11/15 (73.3%) | 211.44 ms / \$0.000023 |
| Cải thiện 2 (Rerank) | Hybrid RRF (top-10) + LLM Rerank (`gpt-4o-mini`), `top_k=3`, `budget=800` | Dùng LLM xếp hạng lại top-10 candidate để đẩy chunk đúng lên #1 | dev | **13/15 (86.7%)** | **15/15 (100.0%)** | **13/15 (86.7%)** | 2336.77 ms / \$0.004102 |
| Phương pháp 6 (Query Transform) | LLM viết lại 2 sub-queries + Hybrid RRF, `top_k=3`, `budget=800` | Mở rộng truy vấn để tăng độ phủ các câu hỏi đa ý | dev | 12/15 (80.0%) | 13/15 (86.7%) | 12/15 (80.0%) | 2685.73 ms / \$0.000836 |
| **Cấu hình đã chọn (Rerank)** | Hybrid RRF + LLM Rerank, `top_k=3`, có filter, `budget=3200` | Đánh giá độc lập sau khi chốt cấu hình trên dev | **test** | **5/5 (100.0%)** | **5/5 (100.0%)** | **5/5 (100.0%)** | 1331.9 ms / \$0.001402 |
| **Baseline (Dense)** | Dense `text-embedding-3-small`, `top_k=3`, có filter, `budget=3200` | Đối chiếu baseline trên cùng tập test | **test** | 3/5 (60.0%) | 5/5 (100.0%) | 5/5 (100.0%) | 208.05 ms / \$0.000010 |

### Phân tích 2 lỗi trên tập `dev` và quá trình cải thiện

1. **Lỗi 1 — Context Budget quá nhỏ (`budget=800` và `1600`) làm rơi bằng chứng thứ 2:**
   - **Hiện tượng:** Khi chạy `bm25` hay `dense` với `budget=800`, `Hit@3` đạt `93.3%–100%` nhưng `Context Evidence Rate` tụt xuống còn `66.7%`. Ở Agent `dense` (`budget=800`), tỷ lệ từ chối nhầm lên tới **11/15** (`q01`, `q03`, `q04`, `q06`, `q08`, `q11`...).
   - **Nguyên nhân:** Hàm `pack_context` chỉ nạp tối đa số chunk vừa khung ký tự. Với chunk dài ~450–600 ký tự, `budget=800` chỉ chứa được đúng 1 chunk, làm mất hoàn toàn chunk thứ 2 chứa vế còn lại của câu hỏi đa ý.
   - **Cách sửa & kết quả:** Tăng `budget` lên `3200` và `top_k=5` kết hợp `rerank` giúp đưa đủ các chunk liên quan vào context.

2. **Lỗi 2 — Định dạng citation khiến `run_agent` từ chối nhầm (`invalid_or_missing_citations`):**
   - **Hiện tượng:** Ở lần chạy `hybrid_rrf` (`budget=1600`) và `rerank` đầu tiên, dù chunk bằng chứng đã nằm trong context (`q01`, `q03`, `q04`, `q06`), `run_agent` vẫn trả về `abstain` với event `invalid_or_missing_citations`.
   - **Nguyên nhân:** LLM sinh citation dạng gộp `[id1, id2]` hoặc kèm tiêu đề `[id1 | file | page]`, không khớp kiểm tra `f"[{cid}]" in answer` trong `starter/agentic.py`.
   - **Cách sửa & kết quả:** Bổ sung hướng dẫn rõ định dạng `[chunk_id_1] [chunk_id_2]` trong `SYSTEM_PROMPT` và bộ chuẩn hóa format citation trong `OpenAIController.decide` (`src/agent.py`). Kết quả trên `dev` giảm từ chối nhầm từ **6/15 (40%)** xuống **0/15 (0%)**.

---

## 3. Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Tool: `Retriever.search(query, filters, top_k)` kết nối các retriever trong `src/retrievers.py`.
  - Điều kiện dừng: Tối đa `max_searches=4` lượt; dừng ngay khi model trả `action="answer"` có đủ citation hợp lệ trong context, hoặc khi model từ chối (`citations=[]`), hoặc chạm ngưỡng `max_searches`.
  - Xử lý lỗi: Lỗi kết nối OpenAI API được bắt riêng và ghi `status="error"` trong `traces.json` để `starter/agent_metrics.py` thống kê tách biệt khỏi chất lượng trả lời.

### Bảng tổng hợp đánh giá Agent (trên `dev` và `test`)

| Cấu hình / split | Tool use | Số lượt gọi (min–max / TB) | Grounded answer 0/1/2 và TB | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| `dense` baseline (`budget=800`, `top_k=3`) / **dev** | 17/17 | 1–4 (TB 2.06) | 0: 11, 1: 0, 2: 4 (**0.53 / 2.0**) | 4/4 (100%) | 4/4 (100%) | 2/2 (100%) | 11/15 (73.3%) |
| `hybrid_rrf` (`budget=1600`, `top_k=3`) / **dev** | 17/17 | 1–4 (TB 1.47) | 0: 6, 1: 4, 2: 5 (**0.93 / 2.0**) | 12/12 (100%) | 9/9 (100%) | 2/2 (100%) | 6/15 (40.0%) |
| `rerank` v2 (`budget=3200`, `top_k=5`) / **dev** | 17/17 | 1–4 (TB 1.35) | 0: 0, 1: 2, 2: 13 (**1.87 / 2.0**) | 26/26 (100%) | 15/15 (100%) | 2/2 (100%) | **0/15 (0.0%)** |
| `dense` baseline (`budget=800`, `top_k=3`) / **test** | 7/7 | 1–4 (TB 3.43) | 0: 3, 1: 2, 2: 0 (**0.40 / 2.0**) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 3/5 (60.0%) |
| **`rerank` v2 (`budget=3200`, `top_k=5`) / test** | **7/7** | **1–4 (TB 1.86)** | **0: 0, 1: 3, 2: 2 (1.40 / 2.0)** | **8/8 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0.0%)** |

### Trích xuất mẫu kiểm tra trên tập `test` (`rerank` v2)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 lượt search (`rerank`, `top_k=5`), trả về đúng `p35_para_05` và `p36_para_04` | "Cường độ tiêu thụ điện trên mỗi 1.000 USD GDP năm 2024 tăng lên 590 kWh so với 579 kWh năm 2023... kim loại cơ bản, kim loại màu, cao su - nhựa tăng lần lượt 9,3%, 16,3% và 23,4%." `[p35_para_05]` `[p36_para_04]` | **Grade 2**; citation `[true, true]`. Đầy đủ cả 2 vế số liệu và đúng nguồn. |
| `q17` | Có đáp án (Lỗi thiếu ý) | 1 lượt search, lấy được `p52_para_03` và `p53_para_02` | "Cần đầu tư 6–9 tỷ USD giai đoạn 2024–2030. Cộng dồn giai đoạn 2031–2050 là 101 tỷ USD theo kịch bản ADS." `[p52_para_03]` `[p53_para_02]` | **Grade 1**; citation `[true, true]`. Đúng giai đoạn 2024–2030 nhưng nhầm số tổng đầu tư cộng dồn ngành điện theo ADS (280 tỷ USD) sang con số khác trong chunk. |
| `q23` | Thiếu bằng chứng | 4 lượt search thử nhiều từ khóa ("kim ngạch xuất khẩu sầu riêng...", "sầu riêng") đều không có thông tin | "Không đủ bằng chứng để trả lời." | **`abstention_correct = true`**. Agent tự thử lại truy vấn nhiều lượt rồi từ chối chính xác, không suy đoán. |

### Đường dẫn trace, hướng dẫn cài đặt, lệnh chạy, thời gian và chi phí
- **Cài đặt môi trường & kết nối:**
  1. Cài thư viện (Python 3.11+):
     ```bash
     pip install openai numpy python-dotenv pypdf pypdfium2 pillow
     ```
  2. Tạo file `.env` từ `.env.example` và điền API key:
     ```env
     OPENAI_API_KEY=sk-...
     OPENAI_EMBED_MODEL=text-embedding-3-small
     OPENAI_CHAT_MODEL=gpt-4o-mini
     ```
- **Lệnh chạy kiểm tra code:** `python -m unittest discover -s tests -v` (15/15 PASS).
- **Lệnh chạy retrieval benchmark:** `python -m src.benchmark_retrieval --split dev --budget 800` và `--split test --budget 3200`.
- **Lệnh chạy Agent & tổng hợp metrics:**
  - `python -m src.run_agent_eval --retriever rerank --split test --budget 3200 --top-k 5 --max-searches 4 --output starter/results/agent_rerank_v2_test`
  - `python starter/agent_metrics.py --cases starter/data/converted/cases.json --traces starter/results/agent_rerank_v2_test/traces.json --reviews starter/results/agent_rerank_v2_test/reviews.json --split test --output starter/results/agent_rerank_v2_test/summary.json`
- **Đường dẫn kết quả:** `starter/results/retrieval_benchmark/`, `starter/results/agent_dense_baseline_dev/`, `starter/results/agent_hybrid_rrf_dev/`, `starter/results/agent_rerank_v2_dev/`, `starter/results/agent_dense_baseline_test/`, `starter/results/agent_rerank_v2_test/`.
- **Thời gian & chi phí LLM:**
  - `agent_dense_baseline_dev`: trung vị `4817.4 ms/câu`, tổng chi phí `\$0.00642`.
  - `agent_rerank_v2_dev`: trung vị `6440.1 ms/câu`, tổng chi phí `\$0.00825`.
  - `agent_dense_baseline_test`: trung vị `7327.3 ms/câu`, tổng chi phí `\$0.00424`.
  - `agent_rerank_v2_test`: trung vị `7259.3 ms/câu`, tổng chi phí `\$0.00426`.

### Đánh đổi và giới hạn
- **Đánh đổi:** `rerank` kết hợp `budget=3200` tăng điểm trung bình Grounded Answer trên `test` từ **0.40** lên **1.40 / 2.0** (và giảm từ chối nhầm từ 60% xuống 0%), nhưng làm độ trễ mỗi lượt retrieval tăng từ ~208 ms (`dense`) lên ~1332 ms (`rerank`) do gọi thêm LLM xếp hạng.
- **Giới hạn:** Tài liệu PDF gốc có một số bảng biểu phức tạp bị mất cấu trúc lưới khi trích text thuần từ Day 6, dẫn đến một số câu hỏi tổng hợp số liệu từ bảng (`q17`, `q19`) dễ bị trích thiếu vế khi thông tin nằm rải rác ở nhiều đoạn khác nhau.


---

## 6. Thử thách nâng cao (+15 điểm cộng) — Multimodal Agentic RAG trên WDR 2026

### 6.1. Tài liệu đã chọn & Quyền truy cập
- **Tài liệu:** *World Development Report 2026: The Promise of Artificial Intelligence* (World Bank Group, **380 trang PDF**).
- **Nguồn chính thức:**
  - Trang xuất bản: https://openknowledge.worldbank.org/entities/publication/8d048ad2-5cc3-4fc0-ba22-2e1f82f5d7de
  - Link tải PDF trực tiếp (tự động tải khi chạy `python -m src.advanced.ingest`): https://openknowledge.worldbank.org/bitstreams/d563aa51-5111-4b9a-b664-552eb0c4ba86/download
- **Giấy phép:** **CC BY 3.0 IGO** (Creative Commons Attribution 3.0 IGO — cho phép chia sẻ và tái tạo tự do với ghi nguồn).

### 6.2. Những gì đã index & những gì bỏ qua (`src/advanced/ingest.py`)
- **Tổng số trang PDF:** `380` trang.
- **Đã index:** `367` trang nội dung thành **1.467 chunks** lưu tại `starter/data/advanced/chunks.json`:
  - **1.363 text chunks** (`chunk_type="text"`): trích xuất text theo trang, tách đoạn có overlap (1.200 ký tự, overlap 150) kèm metadata `pdf_page`, `printed_page`, `section`.
  - **104 visual chunks** (`chunk_type="visual"`): tự động phát hiện 104 trang có Table / Figure / Map / Diagram, render ảnh PNG (`starter/data/advanced/pages/page_XXX.png` qua `pypdfium2`) và dùng OpenAI Vision (`gpt-4o-mini`) trích xuất bảng Markdown chuẩn xác, trục biểu đồ, chú giải và số liệu.
- **Đã bỏ qua (`13` trang):** ghi nhận chi tiết trong `starter/data/advanced/ingest_log.json` gồm 13 trang trắng / trang phân cách chương có 0 ký tự (`PDF pp. 2, 18, 22, 30, 34, 116, 134, 180, 188, 226, 234, 378, 379`).

### 6.3. Kiến trúc Multimodal Agentic RAG (`src/advanced/agent.py`)
- **Retrieval:** Tái sử dụng `OpenAIDenseRetriever` (`text-embedding-3-small`, cache `dense_cache.npz`) kết hợp `BM25Retriever` qua `HybridRRFRetriever` (`rrf_k=60, candidate_k=20`).
- **Agentic Multi-step Planner:** Tối đa `3` vòng tìm kiếm (`gpt-4o-mini`), kiểm tra xem đã đủ bằng chứng cho mọi vế câu hỏi hay chưa; nếu thiếu (ví dụ câu hỏi `multi_hop` nối 2 bảng/biểu đồ ở các chương khác nhau) sẽ tự động sinh truy vấn tìm kiếm bổ sung.
- **Multimodal Vision Synthesis:** Đính kèm tối đa `3` ảnh PNG trang tài liệu truy xuất được (ưu tiên trang có `chunk_type="visual"`) trực tiếp vào bước tổng hợp cuối (`gpt-4o`, `detail="high"`) để đối chiếu số liệu trực quan trên bảng biểu, biểu đồ, bản đồ và sơ đồ trước khi trả lời hoặc từ chối (`abstained: true`).

### 6.4. Benchmark nâng cao (`starter/data/advanced/benchmark.json`)
Tổng **48 câu hỏi** gắn `pdf_page`, `printed_page`, `source` và `evidence` kiểm chứng được từ PDF gốc:
- **`dev` (18 câu):** 15 câu có bằng chứng (`4 table`, `3 diagram_map`, `4 chart`, `2 multi_hop`, `2 text`) + `3 unanswerable`.
- **`test` (30 câu):** `6 table`, `5 diagram_map`, `5 chart`, `4 multi_hop`, `4 text`, `6 unanswerable`.

### 6.5. Tuning trên `dev` & Kết quả trên tập `test` (`src/advanced/evaluate.py`)
| Run | Split | Cấu hình | Pass / Total | Pass Rate | Chi phí (USD) | Latency TB |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `dev_rrf` | `dev` | `hybrid_rrf`, `top_k=5`, `gpt-4o-mini` | 16 / 18 | 88.89% | \$0.14365 | 4.589s |
| `dev_rerank` | `dev` | `hybrid_rerank`, `top_k=7`, `gpt-4o-mini` | 16 / 18 | 88.89% | \$0.16732 | 5.861s |
| `dev_rrf_4o` (chọn) | `dev` | `hybrid_rrf`, `top_k=7`, `gpt-4o` Vision | **17 / 18** | **94.44%** | \$0.28519 | 3.298s |
| **`test_final`** | **`test`** | **`hybrid_rrf`, `top_k=7`, `gpt-4o` Vision (locked)** | **27 / 30** | **90.00%** | **\$0.47257** | **3.663s** |

**Chi tiết tập `test` (30 câu) theo từng nhóm (`starter/results/advanced/test_final.json`):**

| Nhóm câu hỏi | Pass / Total | Pass Rate |
| :--- | :---: | :---: |
| `table` | 6 / 6 | 100.0% |
| `chart` | 5 / 5 | 100.0% |
| `text` | 4 / 4 | 100.0% |
| `diagram_map` | 4 / 5 | 80.0% |
| `multi_hop` | 3 / 4 | 75.0% |
| `unanswerable` | 5 / 6 | 83.33% |
| **Tổng cộng** | **27 / 30** | **90.0%** |

**Phân tích 3 ca lỗi trên tập `test`:**
1. **`adv-test-11` (`diagram_map` — Retrieval miss dẫn đến từ chối nhầm):** Truy vấn không kéo được `PDF p.309` (`Figure 8.1`) mà lấy nhầm các trang thảo luận về procurement phía sau (`pp. 327, 329, 330`), khiến agent chọn `abstained: true` thay vì trả lời.
2. **`adv-test-20` (`multi_hop` — Thiếu chi tiết phụ):** Agent tìm đúng cả 2 trang (`p.120` và `p.137`), trả lời đúng `PROMPTS` (Swahili) và `English 42.6%`, nhưng bỏ qua 2 ý phụ (`small model on its own server` và `Other languages 22.7%`) nên bị chấm strict fail.
3. **`adv-test-25` (`unanswerable` — Lỗi định dạng cờ `abstained`):** Agent trả lời đúng về mặt ngữ nghĩa (*"The study excludes low-income countries, so there is no data available..."*) nhưng đặt `abstained: false` trong JSON, bị rule kiểm tra tự động tính là fail.

### 6.6. Hướng dẫn chạy tái tạo Thử thách Nâng cao
```bash
# 1. Tải (nếu chưa có), trích xuất text + render PNG + chạy Vision index (380 trang)
python -m src.advanced.ingest --workers 8

# 2. Chạy đánh giá trên tập dev hoặc test
python -m src.advanced.evaluate --split dev --mode hybrid_rrf --model gpt-4o --top-k 7 --name dev_rrf_4o
python -m src.advanced.evaluate --split test --mode hybrid_rrf --model gpt-4o --top-k 7 --name test_final
```
