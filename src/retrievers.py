"""All retrieval methods for Day 7: BM25, OpenAI Dense, Hybrid RRF, LLM Rerank, and Query Transform."""
from __future__ import annotations

import json
import math
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

EMBED_COST_PER_1M = 0.02  # text-embedding-3-small
CHAT_IN_COST_PER_1M = 0.15  # gpt-4o-mini input
CHAT_OUT_COST_PER_1M = 0.60  # gpt-4o-mini output


def tokenize(text: str) -> list[str]:
    """Normalize NFC and tokenize Unicode words."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def matches_filter(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Check whether chunk matches all key-value pairs in filters (top-level or metadata)."""
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for k, v in filters.items():
        if v is None:
            continue
        val = chunk.get(k, meta.get(k))
        if val != v:
            return False
    return True


def _format_hit(chunk: dict[str, Any], score: float, strategy: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {**chunk, "score": float(score), "strategy": strategy}
    if extra:
        item.update(extra)
    return item


class BaseRetriever:
    name: str = "base"

    def __init__(self) -> None:
        self.last_cost_usd: float = 0.0
        self.last_meta: dict[str, Any] = {}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    """Okapi BM25 lexical retriever (offline, standard library)."""

    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        n_docs = len(chunks)
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        self.last_cost_usd = 0.0
        self.last_meta = {}
        q_tokens = tokenize(query)
        scored: list[tuple[float, str, int]] = []
        for idx, chunk in enumerate(self.chunks):
            if not matches_filter(chunk, filters):
                continue
            tf_map = self.doc_freqs[idx]
            dl = self.doc_lens[idx]
            score = 0.0
            for term in q_tokens:
                if term not in self.idf:
                    continue
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                score += self.idf[term] * (tf * (self.k1 + 1.0)) / denom
            if score > 0.0:
                scored.append((score, chunk["id"], idx))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [_format_hit(self.chunks[idx], score, self.name) for score, _, idx in scored[:top_k]]


class OpenAIDenseRetriever(BaseRetriever):
    """Dense vector retriever using OpenAI text-embedding-3-small + local numpy cosine index."""

    name = "dense"

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        cache_path: Path | str = "starter/data/converted/embeddings_cache.npz",
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        super().__init__()
        self.chunks = chunks
        self.model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.cache_path = Path(cache_path)
        self.client = client
        self.index_cost_usd = 0.0
        self.index_seconds = 0.0
        self.embeddings = self._load_or_build_index()

    def _get_client(self) -> OpenAI:
        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")
            self.client = OpenAI(api_key=api_key)
        return self.client

    def _load_or_build_index(self) -> np.ndarray:
        chunk_ids = [c["id"] for c in self.chunks]
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            cached_ids = data["ids"].tolist()
            cached_model = str(data["model"].item())
            if cached_ids == chunk_ids and cached_model == self.model:
                vecs = data["embeddings"].astype(np.float32)
                return vecs

        t0 = time.perf_counter()
        client = self._get_client()
        texts = [c["text"] for c in self.chunks]
        all_vecs: list[list[float]] = []
        total_tokens = 0
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            resp = client.embeddings.create(model=self.model, input=batch)
            total_tokens += getattr(resp.usage, "total_tokens", 0) or 0
            all_vecs.extend([d.embedding for d in resp.data])

        mat = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            ids=np.array(chunk_ids),
            model=np.array(self.model),
            embeddings=mat,
        )
        self.index_seconds = time.perf_counter() - t0
        self.index_cost_usd = (total_tokens / 1_000_000) * EMBED_COST_PER_1M
        return mat

    def _embed_query(self, query: str) -> tuple[np.ndarray, float]:
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=[query])
        tokens = getattr(resp.usage, "total_tokens", 0) or 0
        cost = (tokens / 1_000_000) * EMBED_COST_PER_1M
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        return vec / norm, cost

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec, cost = self._embed_query(query)
        self.last_cost_usd = cost
        self.last_meta = {"model": self.model}
        scores = self.embeddings @ q_vec

        mask = np.array([matches_filter(c, filters) for c in self.chunks], dtype=bool)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_scores = scores[valid_indices]
        order = np.argsort(-valid_scores)[:top_k]
        results = []
        for pos in order:
            idx = int(valid_indices[pos])
            results.append(_format_hit(self.chunks[idx], float(scores[idx]), self.name))
        return results


class HybridRRFRetriever(BaseRetriever):
    """Reciprocal Rank Fusion combining Dense and BM25 retrievers."""

    name = "hybrid_rrf"

    def __init__(
        self,
        dense: OpenAIDenseRetriever,
        bm25: BM25Retriever,
        rrf_k: int = 60,
        candidate_k: int = 15,
    ) -> None:
        super().__init__()
        self.dense = dense
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        k_cand = max(self.candidate_k, top_k * 3)
        dense_hits = self.dense.search(query, filters=filters, top_k=k_cand)
        bm25_hits = self.bm25.search(query, filters=filters, top_k=k_cand)
        self.last_cost_usd = self.dense.last_cost_usd + self.bm25.last_cost_usd

        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(dense_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, item in enumerate(bm25_hits, start=1):
            cid = item["id"]
            by_id[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        self.last_meta = {"dense_candidates": len(dense_hits), "bm25_candidates": len(bm25_hits)}
        return [_format_hit(by_id[cid], score, self.name) for cid, score in ordered]


class RerankRetriever(BaseRetriever):
    """Two-stage retrieval: fetch candidates via base retriever, then rerank with OpenAI LLM."""

    name = "rerank"

    def __init__(
        self,
        base_retriever: BaseRetriever,
        candidate_k: int = 10,
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        super().__init__()
        self.base = base_retriever
        self.candidate_k = candidate_k
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.client = client

    def _get_client(self) -> OpenAI:
        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")
            self.client = OpenAI(api_key=api_key)
        return self.client

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidates = self.base.search(query, filters=filters, top_k=max(self.candidate_k, top_k))
        base_cost = self.base.last_cost_usd
        if len(candidates) <= 1:
            self.last_cost_usd = base_cost
            return candidates[:top_k]

        passages = "\n".join(
            f"[{i}] (id={c['id']}, page={c['page']}): {c['text'][:450]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            "Bạn là bộ xếp hạng tài liệu chính xác. Cho câu hỏi và danh sách các đoạn văn bản được đánh số [0], [1], ...\n"
            "Hãy xếp hạng các đoạn văn từ liên quan nhất (chứa bằng chứng trả lời trực tiếp câu hỏi) đến ít liên quan nhất.\n"
            'Trả về JSON đúng định dạng: {"ranked_indices": [0, 2, 1, ...]}\n\n'
            f"Câu hỏi: {query}\n\n"
            f"Các đoạn văn:\n{passages}"
        )
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        usage = resp.usage
        llm_cost = 0.0
        if usage:
            llm_cost = (
                (usage.prompt_tokens / 1_000_000) * CHAT_IN_COST_PER_1M
                + (usage.completion_tokens / 1_000_000) * CHAT_OUT_COST_PER_1M
            )
        self.last_cost_usd = base_cost + llm_cost

        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        indices = parsed.get("ranked_indices", [])
        seen: set[int] = set()
        reranked: list[dict[str, Any]] = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                score = float(len(candidates) - len(reranked))
                reranked.append(_format_hit(candidates[idx], score, self.name))
        for idx, c in enumerate(candidates):
            if idx not in seen:
                reranked.append(_format_hit(c, 0.0, self.name))
        self.last_meta = {"candidate_count": len(candidates), "rerank_model": self.model}
        return reranked[:top_k]


class QueryTransformRetriever(BaseRetriever):
    """Query rewriting/expansion using OpenAI LLM followed by RRF fusion across sub-queries."""

    name = "query_transform"

    def __init__(
        self,
        base_retriever: BaseRetriever,
        rrf_k: int = 60,
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        super().__init__()
        self.base = base_retriever
        self.rrf_k = rrf_k
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.client = client

    def _get_client(self) -> OpenAI:
        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")
            self.client = OpenAI(api_key=api_key)
        return self.client

    def _rewrite(self, query: str) -> tuple[list[str], float]:
        prompt = (
            "Hãy viết lại và mở rộng câu hỏi sau thành 2 truy vấn tìm kiếm ngắn gọn bằng tiếng Việt "
            "giữ nguyên các từ khóa quan trọng, số liệu, thuật ngữ chuyên ngành để tìm kiếm hiệu quả trong báo cáo kinh tế/xe điện.\n"
            'Trả về JSON đúng định dạng: {"queries": ["truy vấn 1", "truy vấn 2"]}\n\n'
            f"Câu hỏi gốc: {query}"
        )
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        usage = resp.usage
        cost = 0.0
        if usage:
            cost = (
                (usage.prompt_tokens / 1_000_000) * CHAT_IN_COST_PER_1M
                + (usage.completion_tokens / 1_000_000) * CHAT_OUT_COST_PER_1M
            )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        rewrites = [q.strip() for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()]
        return [query] + rewrites[:2], cost

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        queries, rewrite_cost = self._rewrite(query)
        total_cost = rewrite_cost
        rrf_scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        per_q_k = max(top_k * 3, 8)
        for q in queries:
            hits = self.base.search(q, filters=filters, top_k=per_q_k)
            total_cost += self.base.last_cost_usd
            for rank, item in enumerate(hits, start=1):
                cid = item["id"]
                by_id[cid] = item
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        self.last_cost_usd = total_cost
        self.last_meta = {"rewritten_queries": queries}
        ordered = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [_format_hit(by_id[cid], score, self.name) for cid, score in ordered]
