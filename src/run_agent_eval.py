"""Run OpenAI Agentic RAG on Day 7 cases and save traces compatible with starter/agent_metrics.py."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))
from agentic import run_agent  # noqa: E402

from .agent import OpenAIController  # noqa: E402
from .retrievers import (  # noqa: E402
    BM25Retriever,
    HybridRRFRetriever,
    OpenAIDenseRetriever,
    QueryTransformRetriever,
    RerankRetriever,
)


def build_retriever(method: str, chunks: list[dict[str, Any]]) -> Any:
    bm25 = BM25Retriever(chunks)
    if method == "bm25":
        return bm25
    dense = OpenAIDenseRetriever(chunks)
    if method == "dense":
        return dense
    hybrid = HybridRRFRetriever(dense=dense, bm25=bm25, candidate_k=15)
    if method == "hybrid_rrf":
        return hybrid
    if method == "rerank":
        return RerankRetriever(base_retriever=hybrid, candidate_k=10)
    if method == "query_transform":
        return QueryTransformRetriever(base_retriever=hybrid)
    raise ValueError(f"Unsupported retriever method: {method}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluation on Day 7 cases")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--cases", type=Path, default=Path("starter/data/converted/cases.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid_rrf", "rerank", "query_transform"],
        default="hybrid_rrf",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1600)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    cases = [c for c in json.loads(args.cases.read_text(encoding="utf-8")) if c["split"] == args.split]
    q_map = {q["id"]: q for q in json.loads(args.questions.read_text(encoding="utf-8"))}

    retriever = build_retriever(args.retriever, chunks)
    controller = OpenAIController(model=args.model)

    out_dir = args.output or Path(f"starter/results/agent_{args.retriever}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: list[dict[str, Any]] = []
    review_templates: list[dict[str, Any]] = []
    total_cost = 0.0

    for c in cases:
        qid = c["id"]
        q_meta = q_map.get(qid, {})
        flt = q_meta.get("filter")
        controller.reset_metrics()
        t0 = time.perf_counter()
        try:
            res = run_agent(
                c["question"],
                controller,
                retriever,
                filters=flt,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            search_events = [e for e in res["events"] if e.get("event") == "search"]
            trace_item = {
                "query_id": qid,
                "split": c["split"],
                "question": c["question"],
                "answerable": c["answerable"],
                "status": res["status"],
                "answer": res["answer"],
                "citations": res["citations"],
                "events": res["events"],
                "search_calls": len(search_events),
                "elapsed_ms": round(elapsed_ms, 2),
                "cost_usd": round(controller.total_cost_usd, 6),
                "prompt_tokens": controller.total_prompt_tokens,
                "completion_tokens": controller.total_completion_tokens,
            }
            total_cost += controller.total_cost_usd

            if c["answerable"]:
                review_templates.append(
                    {
                        "query_id": qid,
                        "expected_answer": q_meta.get("expected_answer", ""),
                        "agent_answer": res["answer"],
                        "citations": res["citations"],
                        "citation_support": [True] * len(res["citations"]),
                        "answer_grade": 2 if res["status"] == "answered" else 0,
                        "notes": "TODO: đối chiếu từng citation với context gốc trước khi chốt điểm",
                    }
                )
            else:
                review_templates.append(
                    {
                        "query_id": qid,
                        "agent_status": res["status"],
                        "agent_answer": res["answer"],
                        "citation_support": [],
                        "abstention_correct": res["status"] == "abstain",
                        "notes": "TODO: kiểm tra agent thực sự từ chối và không suy đoán",
                    }
                )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            trace_item = {
                "query_id": qid,
                "split": c["split"],
                "question": c["question"],
                "answerable": c["answerable"],
                "status": "error",
                "error": str(exc),
                "answer": "",
                "citations": [],
                "events": [],
                "elapsed_ms": round(elapsed_ms, 2),
                "cost_usd": round(controller.total_cost_usd, 6),
            }
        traces.append(trace_item)
        print(f"[{qid}] status={trace_item['status']} searches={trace_item.get('search_calls', 0)} ms={trace_item['elapsed_ms']}")

    (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "reviews_draft.json").write_text(
        json.dumps(review_templates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved traces to {out_dir / 'traces.json'} (total LLM cost: ${total_cost:.5f})")
    print(f"Draft reviews saved to {out_dir / 'reviews_draft.json'} -> verify manually, save as reviews.json, then run starter/agent_metrics.py")


if __name__ == "__main__":
    main()
