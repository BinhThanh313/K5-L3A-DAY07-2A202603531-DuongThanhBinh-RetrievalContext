"""Benchmark all retrieval methods on Day 7 dev or test split."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter"))
from baseline import TfidfIndex, hit, pack_context  # noqa: E402

from .retrievers import (  # noqa: E402
    BM25Retriever,
    HybridRRFRetriever,
    OpenAIDenseRetriever,
    QueryTransformRetriever,
    RerankRetriever,
)


class _TfidfWrapper:
    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)
        self.last_cost_usd = 0.0
        self.last_meta: dict[str, Any] = {}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        # TfidfIndex does not support filters; keep baseline behavior
        return self.index.search(query, top_k=top_k)


def evaluate_method(
    name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    use_filter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_cost = 0.0
    hit1 = 0
    hit3 = 0
    context_hit = 0
    mrr_sum = 0.0

    for q in questions:
        flt = q.get("filter") if use_filter else None
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=flt, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        q_cost = float(getattr(retriever, "last_cost_usd", 0.0))
        total_cost += q_cost

        context, ids = pack_context(results, budget=budget)
        marked = [{**c, "hit": hit(c, q["gold"])} for c in results]
        top3 = marked[:3]
        h1 = bool(top3 and top3[0]["hit"])
        h3 = any(c["hit"] for c in top3)
        ctx_ok = any(c["hit"] for c in marked if c["id"] in ids)
        rr = 0.0
        for rank, c in enumerate(marked, start=1):
            if c["hit"]:
                rr = 1.0 / rank
                break

        hit1 += int(h1)
        hit3 += int(h3)
        context_hit += int(ctx_ok)
        mrr_sum += rr

        traces.append(
            {
                "query_id": q["id"],
                "split": q["split"],
                "question": q["question"],
                "strategy": name,
                "filter": flt,
                "top_k": top_k,
                "budget_chars": budget,
                "hit1": h1,
                "hit3": h3,
                "context_hit": ctx_ok,
                "rr": rr,
                "elapsed_ms": round(elapsed_ms, 2),
                "cost_usd": round(q_cost, 6),
                "meta": getattr(retriever, "last_meta", {}),
                "context_ids": ids,
                "results": marked,
            }
        )

    n = len(questions)
    summary = {
        "strategy": name,
        "n": n,
        "hit1": f"{hit1}/{n} ({hit1 / n * 100:.1f}%)" if n else "0/0",
        "hit3": f"{hit3}/{n} ({hit3 / n * 100:.1f}%)" if n else "0/0",
        "context_hit": f"{context_hit}/{n} ({context_hit / n * 100:.1f}%)" if n else "0/0",
        "mrr": round(mrr_sum / n, 4) if n else 0.0,
        "median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "query_cost_usd": round(total_cost, 6),
    }
    return summary, traces


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Benchmark all retrieval methods on Day 7 dataset")
    parser.add_argument("--chunks", type=Path, default=Path("starter/data/converted/chunks.json"))
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=["tfidf", "bm25", "dense", "hybrid_rrf", "rerank", "query_transform"])
    parser.add_argument("--output", type=Path, default=Path("starter/results/retrieval_benchmark"))
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q["split"] == args.split]
    if not questions:
        raise ValueError(f"No questions found for split={args.split}")

    need_openai = any(m in args.methods for m in ("dense", "hybrid_rrf", "rerank", "query_transform"))
    bm25 = BM25Retriever(chunks)
    dense = OpenAIDenseRetriever(chunks) if need_openai else None

    registry: dict[str, Any] = {
        "tfidf": _TfidfWrapper(chunks),
        "bm25": bm25,
    }
    if dense is not None:
        hybrid = HybridRRFRetriever(dense=dense, bm25=bm25, candidate_k=15)
        registry["dense"] = dense
        registry["hybrid_rrf"] = hybrid
        registry["rerank"] = RerankRetriever(base_retriever=hybrid, candidate_k=10)
        registry["query_transform"] = QueryTransformRetriever(base_retriever=hybrid)

    out_dir = args.output / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []

    for m in args.methods:
        if m not in registry:
            raise ValueError(f"Unknown method: {m}")
        summary, traces = evaluate_method(
            m, registry[m], questions, top_k=args.top_k, budget=args.budget
        )
        all_summaries.append(summary)
        (out_dir / f"traces_{m}.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== Retrieval Benchmark ({args.split}, n={len(questions)}, budget={args.budget}) ===")
    print(f"{'Method':<18} {'Hit@1':<15} {'Hit@3':<15} {'CtxHit':<15} {'MRR':<8} {'Med(ms)':<10} {'Cost($)'}")
    print("-" * 92)
    for s in all_summaries:
        print(
            f"{s['strategy']:<18} {s['hit1']:<15} {s['hit3']:<15} {s['context_hit']:<15} "
            f"{s['mrr']:<8} {s['median_ms']:<10} {s['query_cost_usd']}"
        )


if __name__ == "__main__":
    main()
