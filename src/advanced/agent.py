"""Multimodal Agentic RAG for WDR 2026 (Advanced Challenge).

Combines Hybrid RRF / Rerank retrieval over text + Vision-indexed chunks
with iterative agentic search and direct PNG page inspection via OpenAI Vision.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.retrievers import (
    BM25Retriever,
    HybridRRFRetriever,
    OpenAIDenseRetriever,
    RerankRetriever,
)

ROOT = Path(__file__).resolve().parents[2]
ADV_DIR = ROOT / "starter" / "data" / "advanced"
CHUNKS_PATH = ADV_DIR / "chunks.json"
DENSE_CACHE_PATH = ADV_DIR / "dense_cache.npz"

PRICING_USD_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def load_advanced_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    for c in chunks:
        if "text" not in c:
            c["text"] = f"{c.get('title', '')}\n{c.get('content', '')}".strip()
        if "source" not in c:
            c["source"] = f"WDR2026_p{c.get('pdf_page', 0)}"
        if "page" not in c:
            c["page"] = c.get("pdf_page", 0)
    return chunks


def build_advanced_retriever(chunks: list[dict], mode: str = "hybrid_rrf"):
    dense = OpenAIDenseRetriever(chunks, cache_path=DENSE_CACHE_PATH)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRRFRetriever(dense, bm25, rrf_k=60, candidate_k=20)
    if mode == "hybrid_rrf":
        return hybrid
    if mode == "hybrid_rerank":
        return RerankRetriever(hybrid, candidate_k=12)
    raise ValueError(f"Unknown retriever mode: {mode}")


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = PRICING_USD_PER_1M.get(model, PRICING_USD_PER_1M["gpt-4o-mini"])
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000.0


def _pack_context(chunks: list[dict], budget_chars: int = 8000) -> str:
    parts: list[str] = []
    used = 0
    for c in chunks:
        header = (
            f"[{c['id']}] (PDF p.{c['pdf_page']}, printed p.{c['printed_page']}, "
            f"type={c['chunk_type']}) {c['title']}"
        )
        body = c["content"].strip()
        block = f"{header}\n{body}\n"
        if used + len(block) > budget_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts)


def run_multimodal_agent(
    question: str,
    retriever,
    chunk_by_id: dict[str, dict],
    *,
    client: OpenAI | None = None,
    model: str = "gpt-4o-mini",
    max_searches: int = 3,
    top_k: int = 5,
    budget_chars: int = 12000,
    max_images: int = 3,
) -> dict:
    if client is None:
        load_dotenv(ROOT / ".env", override=True)
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    t0 = time.perf_counter()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    seen_ids: set[str] = set()
    collected: list[dict] = []
    search_trace: list[dict] = []

    current_query = question
    for step in range(1, max_searches + 1):
        raw_hits = retriever.search(current_query, top_k=top_k)
        hits = [chunk_by_id[h["id"]] for h in raw_hits if h["id"] in chunk_by_id]
        new_hits = []
        for h in hits:
            if h["id"] not in seen_ids:
                seen_ids.add(h["id"])
                collected.append(h)
                new_hits.append(h["id"])
        search_trace.append({"step": step, "query": current_query, "new_chunk_ids": new_hits})

        if step == max_searches:
            break

        # Decide whether another search is needed (especially for multi-hop questions)
        ctx_preview = _pack_context(collected, budget_chars=4500)
        plan_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a retrieval planner for World Development Report 2026. "
                        "Given a question and retrieved chunks, check if EVERY part of the question "
                        "is explicitly answered (or if the question is clearly outside WDR 2026). "
                        "If ANY sub-question, figure, table, or named concept is still missing, "
                        "issue a targeted follow-up search query for that missing piece. "
                        'Return JSON: {"action": "answer"} or {"action": "search", "query": "..."}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nRetrieved so far:\n{ctx_preview}",
                },
            ],
        )
        if plan_resp.usage:
            total_prompt_tokens += plan_resp.usage.prompt_tokens
            total_completion_tokens += plan_resp.usage.completion_tokens

        try:
            plan = json.loads(plan_resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            plan = {"action": "answer"}

        if plan.get("action") != "search" or not plan.get("query"):
            break
        current_query = str(plan["query"]).strip()

    # Select up to max_images unique page PNGs from collected chunks (prioritize visual chunks)
    visual_first = sorted(
        collected,
        key=lambda c: (0 if c.get("chunk_type") == "visual" else 1),
    )
    attached_images: list[str] = []
    seen_imgs: set[str] = set()
    for c in visual_first:
        rel = c.get("image_path")
        if rel and rel not in seen_imgs:
            abs_p = ROOT / rel
            if abs_p.exists():
                seen_imgs.add(rel)
                attached_images.append(rel)
                if len(attached_images) >= max_images:
                    break

    context_text = _pack_context(collected, budget_chars=budget_chars)
    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Question: {question}\n\n"
                f"Retrieved Context from WDR 2026:\n{context_text}\n\n"
                "Instructions:\n"
                "- Answer ONLY using the retrieved context and attached page images.\n"
                "- Cite specific PDF page(s) and Figure/Table/Map numbers when applicable.\n"
                "- If the context and images do NOT contain the requested facts, numbers, "
                "or scope (e.g., missing country/year/metric not in WDR 2026), you MUST abstain: "
                'set "abstained": true and state clearly what is missing.\n'
                'Return JSON: {"answer": "...", "abstained": false, "citations": ["p.XX"]}'
            ),
        }
    ]
    for rel in attached_images:
        b64 = base64.b64encode((ROOT / rel).read_bytes()).decode("utf-8")
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            }
        )

    ans_resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=700,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise multimodal research assistant for the World Bank "
                    "World Development Report 2026. Never fabricate numbers or facts."
                ),
            },
            {"role": "user", "content": user_content},
        ],
    )
    if ans_resp.usage:
        total_prompt_tokens += ans_resp.usage.prompt_tokens
        total_completion_tokens += ans_resp.usage.completion_tokens

    raw = ans_resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"answer": raw, "abstained": False, "citations": []}

    latency_s = round(time.perf_counter() - t0, 3)
    cost_usd = round(_estimate_cost(model, total_prompt_tokens, total_completion_tokens), 6)

    return {
        "answer": str(parsed.get("answer", "")).strip(),
        "abstained": bool(parsed.get("abstained", False)),
        "citations": parsed.get("citations", []),
        "searches_used": len(search_trace),
        "search_trace": search_trace,
        "retrieved_pages": sorted({c["pdf_page"] for c in collected}),
        "attached_images": attached_images,
        "latency_s": latency_s,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "cost_usd": cost_usd,
    }
