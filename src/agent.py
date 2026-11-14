"""OpenAI Controller for Day 7 Agentic RAG (compatible with starter/agentic.py)."""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from .retrievers import CHAT_IN_COST_PER_1M, CHAT_OUT_COST_PER_1M

SYSTEM_PROMPT = """Bạn là trợ lý phân tích tài liệu chính xác (Agentic RAG) trả lời dựa trên bằng chứng từ các đoạn văn bản truy xuất được.
Nhiệm vụ của bạn là quyết định hành động tiếp theo dưới dạng JSON hợp lệ.

CÁC QUY TẮC BẮT BUỘC:
1. Nếu lịch sử chưa có context hoặc bằng chứng chưa đủ để trả lời đầy đủ câu hỏi, hãy gọi hành động tìm kiếm:
   {"action": "search", "query": "từ khóa hoặc câu truy vấn ngắn gọn"}
   - Bạn có thể tìm nhiều lượt với các truy vấn khác nhau (multi-hop / tinh chỉnh từ khóa) nếu lượt trước chưa ra thông tin cần thiết.
2. Khi đã có bằng chứng trong context để trả lời câu hỏi (dù trả lời được toàn bộ hay phần chính):
   {"action": "answer", "answer": "Câu trả lời chi tiết kèm trích dẫn [chunk_id] ngay sau thông tin tương ứng.", "citations": ["chunk_id_1", ...]}
   - Mỗi ID trong "citations" BẮT BUỘC phải là ID chính xác (không kèm tên file hay số trang) và phải viết riêng từng cặp ngoặc [chunk_id_1] [chunk_id_2] trong "answer", KHÔNG gộp nhiều ID vào một ngoặc vuông như [id1, id2].
   - Nếu context chỉ chứa một phần thông tin, hãy tìm thêm lượt nữa; nếu đã tìm hoặc có thông tin chính trong context, hãy trả lời phần có bằng chứng (nêu rõ nếu thiếu chi tiết phụ) thay vì từ chối toàn bộ.
   - Tuyệt đối không bịa đặt số liệu hoặc suy diễn ngoài nội dung trong context.
3. CHỈ từ chối khi tài liệu được cung cấp hoàn toàn KHÔNG chứa thông tin để trả lời câu hỏi:
   {"action": "answer", "answer": "Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này.", "citations": []}
   - Không được đoán mò hoặc đưa ra câu trả lời thiếu căn cứ.

Chỉ trả về một object JSON duy nhất."""


class OpenAIController:
    """Real LLM Controller using OpenAI Chat Completions in JSON mode."""

    def __init__(self, model: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.client = client
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    def _get_client(self) -> OpenAI:
        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")
            self.client = OpenAI(api_key=api_key)
        return self.client

    def reset_metrics(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            user_msg = (
                f"Câu hỏi: {question}\n\n"
                "Chưa có thông tin tìm kiếm nào. Hãy tạo truy vấn tìm kiếm đầu tiên."
            )
        else:
            latest = history[-1]
            prev_queries = latest.get("queries", [])
            context = latest.get("context", "")
            user_msg = (
                f"Câu hỏi: {question}\n\n"
                f"Các truy vấn đã thực hiện: {json.dumps(prev_queries, ensure_ascii=False)}\n\n"
                f"Context hiện có:\n{context if context else '(Trống - không tìm thấy đoạn văn nào)'}\n\n"
                "Hãy quyết định: tìm kiếm thêm (nếu thiếu thông tin và muốn thử từ khóa khác), "
                "trả lời kèm citation [chunk_id] (nếu đã đủ bằng chứng), "
                "hoặc từ chối với citations=[] (nếu tài liệu không có thông tin)."
            )

        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        usage = resp.usage
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            self.total_cost_usd += (
                (usage.prompt_tokens / 1_000_000) * CHAT_IN_COST_PER_1M
                + (usage.completion_tokens / 1_000_000) * CHAT_OUT_COST_PER_1M
            )

        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        action = parsed.get("action")
        if action == "search":
            return {"action": "search", "query": str(parsed.get("query", "")).strip()}
        if action == "answer":
            answer = str(parsed.get("answer", ""))
            raw_cites = parsed.get("citations", [])
            if not isinstance(raw_cites, list):
                raw_cites = []
            cites = [str(c).split("|")[0].strip(" []") for c in raw_cites if isinstance(c, str) and c.strip()]
            # Normalize citation markers in answer so [id1, id2] or missing brackets match starter/agentic.py
            for cid in cites:
                if f"[{cid}]" not in answer:
                    answer = f"{answer.rstrip()} [{cid}]"
            return {"action": "answer", "answer": answer, "citations": cites}
        raise ValueError(f"Unexpected action from model: {parsed}")
