"""Evaluate Multimodal Agentic RAG on Advanced Benchmark (strict pass/fail).

Usage:
    python -m src.advanced.evaluate --split dev --mode hybrid_rrf --name dev_rrf
    python -m src.advanced.evaluate --split test --mode hybrid_rerank --name test_final
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.advanced.agent import (
    build_advanced_retriever,
    load_advanced_chunks,
    run_multimodal_agent,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH_PATH = ROOT / "starter" / "data" / "advanced" / "benchmark.json"
OUT_DIR = ROOT / "starter" / "results" / "advanced"


def judge_answer(client: OpenAI, case: dict, pred: dict) -> tuple[bool, str]:
    is_unans = case["category"] == "unanswerable"
    if is_unans:
        if pred["abstained"]:
            return True, "Correctly abstained on unanswerable question."
        return False, f"Failed to abstain on unanswerable question. Answer: {pred['answer'][:160]}"

    if pred["abstained"]:
        return False, f"Falsely abstained on answerable question. Answer: {pred['answer'][:160]}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=180,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict benchmark grader. Grade PASS (true) if the model's answer "
                    "accurately includes all key facts, numbers, and relationships required by the "
                    "expected_answer without contradicting the evidence or fabricating details. "
                    "Do NOT penalize the model for including additional true, relevant details from the document "
                    "beyond the short expected_answer. "
                    'Return JSON: {"pass": true/false, "reason": "..."}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {case['question']}\n"
                    f"Expected Answer: {case['expected_answer']}\n"
                    f"Ground-Truth Evidence: {case['evidence']}\n"
                    f"Model Answer: {pred['answer']}\n"
                    f"Model Citations: {pred['citations']}"
                ),
            },
        ],
    )
    try:
        j = json.loads(resp.choices[0].message.content or "{}")
        return bool(j.get("pass", False)), str(j.get("reason", ""))
    except json.JSONDecodeError:
        return False, "Judge JSON parse error"


def run_eval(
    split: str,
    mode: str,
    run_name: str,
    model: str = "gpt-4o-mini",
    max_searches: int = 3,
    top_k: int = 5,
    max_images: int = 3,
) -> dict:
    load_dotenv(ROOT / ".env", override=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    cases = [c for c in json.loads(BENCH_PATH.read_text(encoding="utf-8")) if c["split"] == split]
    chunks = load_advanced_chunks()
    chunk_by_id = {c["id"]: c for c in chunks}
    retriever = build_advanced_retriever(chunks, mode=mode)

    results: list[dict] = []
    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "total": 0})

    for idx, case in enumerate(cases, 1):
        pred = run_multimodal_agent(
            case["question"],
            retriever,
            chunk_by_id,
            client=client,
            model=model,
            max_searches=max_searches,
            top_k=top_k,
            max_images=max_images,
        )
        passed, reason = judge_answer(client, case, pred)
        cat = case["category"]
        cat_stats[cat]["total"] += 1
        if passed:
            cat_stats[cat]["pass"] += 1

        results.append(
            {
                "id": case["id"],
                "category": cat,
                "question": case["question"],
                "expected": case.get("expected_answer") or case.get("expected_behavior"),
                "target_page": case.get("page"),
                "passed": passed,
                "judge_reason": reason,
                **pred,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"[{idx}/{len(cases)}] {case['id']} ({cat}) -> {status} | pages={pred['retrieved_pages'][:5]} | {pred['latency_s']}s")

    total_pass = sum(1 for r in results if r["passed"])
    total_n = len(results)
    summary = {
        "run_name": run_name,
        "split": split,
        "mode": mode,
        "model": model,
        "max_searches": max_searches,
        "top_k": top_k,
        "max_images": max_images,
        "total": total_n,
        "passed": total_pass,
        "pass_rate": round(total_pass / total_n, 4) if total_n else 0.0,
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 5),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / total_n, 3) if total_n else 0.0,
        "by_category": {
            k: {
                "passed": v["pass"],
                "total": v["total"],
                "pass_rate": round(v["pass"] / v["total"], 4),
            }
            for k, v in sorted(cat_stats.items())
        },
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{run_name}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--mode", choices=["hybrid_rrf", "hybrid_rerank"], default="hybrid_rrf")
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-searches", type=int, default=3)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-images", type=int, default=3)
    args = p.parse_args()
    run_eval(
        split=args.split,
        mode=args.mode,
        run_name=args.name,
        model=args.model,
        max_searches=args.max_searches,
        top_k=args.top_k,
        max_images=args.max_images,
    )
