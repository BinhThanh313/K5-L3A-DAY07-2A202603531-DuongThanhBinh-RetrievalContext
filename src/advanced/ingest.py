"""Ingest WDR 2026 (380 pages) into text + multimodal visual chunks.

Usage:
    python -m src.advanced.ingest
    python -m src.advanced.ingest --skip-vision   # text + PNG render only
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import urllib.request
from pathlib import Path

import pypdf
import pypdfium2 as pdfium
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
ADV_DIR = ROOT / "starter" / "data" / "advanced"
PDF_PATH = ADV_DIR / "wdr2026.pdf"
PAGES_DIR = ADV_DIR / "pages"
CHUNKS_PATH = ADV_DIR / "chunks.json"
LOG_PATH = ADV_DIR / "ingest_log.json"
VISION_CACHE_PATH = ADV_DIR / "vision_cache.json"

WDR2026_URL = (
    "https://openknowledge.worldbank.org/bitstreams/"
    "d563aa51-5111-4b9a-b664-552eb0c4ba86/download"
)

SECTION_RULES = [
    (1, 33, "Front Matter"),
    (35, 76, "Overview"),
    (77, 140, "Part 1: Foundations & AI Stack (Ch 1-3, Spotlight 1)"),
    (141, 266, "Part 2: Impacts & Pathways (Ch 4-6, Spotlights 2-4)"),
    (267, 377, "Part 3: Policy & Governance (Ch 7-9, Spotlights 5-7)"),
    (380, 380, "Back Cover"),
]


def _get_section(pdf_page: int) -> str:
    for start, end, label in SECTION_RULES:
        if start <= pdf_page <= end:
            return label
    return "Other"


def _printed_page(pdf_page: int) -> str:
    if pdf_page >= 35 and pdf_page <= 377:
        return str(pdf_page - 34)
    return f"front-{pdf_page}"


def ensure_pdf(pdf_path: Path = PDF_PATH) -> Path:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists() and pdf_path.stat().st_size > 1_000_000:
        return pdf_path
    print(f"Downloading WDR 2026 PDF to {pdf_path}...")
    req = urllib.request.Request(WDR2026_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(pdf_path, "wb") as f:
        f.write(resp.read())
    return pdf_path


def _split_text_chunks(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    clean = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]
    paragraphs = clean.split("\n\n")
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 2 <= max_chars:
            cur = f"{cur}\n\n{para}".strip()
        else:
            if cur:
                chunks.append(cur)
                tail = cur[-overlap:] if len(cur) > overlap else cur
                cur = f"{tail}\n\n{para}".strip()
            else:
                for i in range(0, len(para), max_chars - overlap):
                    chunks.append(para[i : i + max_chars])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def _detect_visual_tags(text: str, pdf_page: int) -> list[str]:
    if pdf_page < 35 or pdf_page > 377:
        return []
    tags: list[str] = []
    for kind, pattern in [
        ("table", r"(?m)^Table\s+[O0-9PS]\d*[\.\d]*|\bTable\s+(?:O\.|[1-9]\.|S\d|P\d)"),
        ("figure", r"(?m)^Figure\s+[O0-9PSB]\d*[\.\d]*|\bFigure\s+(?:O\.|[1-9]\.|S\d|P\d|B\d)"),
        ("map", r"(?m)^Map\s+[O0-9]\d*[\.\d]*|\bMap\s+(?:O\.|[1-9]\.)"),
    ]:
        if re.search(pattern, text):
            tags.append(kind)
    return tags


def render_visual_pages(pdf_path: Path, visual_pages: list[int], scale: float = 1.5) -> dict[int, Path]:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    rendered: dict[int, Path] = {}
    for pno in visual_pages:
        out_path = PAGES_DIR / f"page_{pno:03d}.png"
        if not out_path.exists():
            page = doc[pno - 1]
            bitmap = page.render(scale=scale)
            pil_img = bitmap.to_pil()
            pil_img.save(out_path, format="PNG", optimize=True)
        rendered[pno] = out_path
    doc.close()
    return rendered


def _extract_vision_one(
    client: OpenAI,
    model: str,
    pdf_page: int,
    img_path: Path,
    tags: list[str],
) -> str:
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    prompt = (
        f"You are indexing page {pdf_page} (printed page {_printed_page(pdf_page)}) of "
        f"World Development Report 2026: The Promise of Artificial Intelligence.\n"
        f"Detected visual elements: {', '.join(tags)}.\n"
        "Extract ALL visual information accurately:\n"
        "1. State the exact Figure / Table / Map / Box number and full title.\n"
        "2. For any TABLE: transcribe the complete table in clean Markdown with exact headers, rows, values, units, and notes/sources.\n"
        "3. For any CHART / GRAPH / MAP / DIAGRAM: describe the chart type, axes, units, legend, key categories/countries, exact labeled data points/values, trends, and source.\n"
        "4. Be concise, factual, and preserve all numbers exactly as printed. Do not invent values."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1400,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def run_ingest(skip_vision: bool = False, vision_model: str = "gpt-4o-mini", max_workers: int = 6) -> dict:
    load_dotenv(ROOT / ".env", override=True)
    pdf_path = ensure_pdf()
    reader = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    skipped_pages: list[dict] = []
    visual_page_tags: dict[int, list[str]] = {}
    page_texts: dict[int, str] = {}

    for pno, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if len(text) < 40:
            skipped_pages.append(
                {
                    "pdf_page": pno,
                    "printed_page": _printed_page(pno),
                    "reason": f"blank_or_divider_page ({len(text)} chars)",
                }
            )
            continue
        page_texts[pno] = text
        tags = _detect_visual_tags(text, pno)
        if tags:
            visual_page_tags[pno] = tags

    visual_pages = sorted(visual_page_tags.keys())
    rendered = render_visual_pages(pdf_path, visual_pages)

    vision_cache: dict[str, str] = {}
    if VISION_CACHE_PATH.exists():
        vision_cache = json.loads(VISION_CACHE_PATH.read_text(encoding="utf-8"))

    if not skip_vision:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing in .env; cannot run Vision extraction.")
        client = OpenAI(api_key=api_key)
        todo = [p for p in visual_pages if str(p) not in vision_cache]
        if todo:
            print(f"Running OpenAI Vision ({vision_model}) on {len(todo)} visual pages...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_map = {
                    pool.submit(
                        _extract_vision_one, client, vision_model, p, rendered[p], visual_page_tags[p]
                    ): p
                    for p in todo
                }
                done_cnt = 0
                for fut in concurrent.futures.as_completed(fut_map):
                    p = fut_map[fut]
                    vision_cache[str(p)] = fut.result()
                    done_cnt += 1
                    if done_cnt % 15 == 0 or done_cnt == len(todo):
                        print(f"  Vision extracted {done_cnt}/{len(todo)} pages")
                        VISION_CACHE_PATH.write_text(
                            json.dumps(vision_cache, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
            VISION_CACHE_PATH.write_text(
                json.dumps(vision_cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    chunks: list[dict] = []
    for pno in sorted(page_texts.keys()):
        section = _get_section(pno)
        printed = _printed_page(pno)
        img_rel = f"starter/data/advanced/pages/page_{pno:03d}.png" if pno in rendered else None

        for idx, piece in enumerate(_split_text_chunks(page_texts[pno]), 1):
            chunks.append(
                {
                    "id": f"p{pno:03d}_t{idx}",
                    "title": f"WDR 2026 | PDF p.{pno} (printed p.{printed}) | {section}",
                    "content": piece,
                    "pdf_page": pno,
                    "printed_page": printed,
                    "section": section,
                    "chunk_type": "text",
                    "visual_tags": visual_page_tags.get(pno, []),
                    "image_path": img_rel,
                }
            )

        if str(pno) in vision_cache and vision_cache[str(pno)]:
            chunks.append(
                {
                    "id": f"p{pno:03d}_vis",
                    "title": f"WDR 2026 Visuals ({', '.join(visual_page_tags.get(pno, []))}) | PDF p.{pno} (printed p.{printed})",
                    "content": vision_cache[str(pno)],
                    "pdf_page": pno,
                    "printed_page": printed,
                    "section": section,
                    "chunk_type": "visual",
                    "visual_tags": visual_page_tags.get(pno, []),
                    "image_path": img_rel,
                }
            )

    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "total_pdf_pages": total_pages,
        "indexed_pages": len(page_texts),
        "skipped_pages_count": len(skipped_pages),
        "skipped_pages": skipped_pages,
        "visual_pages_count": len(visual_pages),
        "visual_pages": visual_pages,
        "total_chunks": len(chunks),
        "text_chunks": sum(1 for c in chunks if c["chunk_type"] == "text"),
        "visual_chunks": sum(1 for c in chunks if c["chunk_type"] == "visual"),
    }
    LOG_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("skipped_pages", "visual_pages")}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--vision-model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run_ingest(skip_vision=args.skip_vision, vision_model=args.vision_model, max_workers=args.workers)
