"""Stage 3 - ENRICH: make non-text content searchable, and give every chunk context.

The embedding model only understands text. So:

  * Images -> Claude (vision) writes a retrieval-oriented description: what the figure shows,
              every label and number it can read, and the takeaway. That description is embedded;
              the original PNG is kept on disk and handed to the agent at answer time.
              (This is the "multi-vector" / "summary-index" pattern: search on a summary,
              answer from the original.)
  * Tables -> Claude writes a short summary (what the table is about, notable values). Raw
              Markdown tables embed poorly because they are mostly pipes and numbers; the
              summary carries the meaning, the Markdown carries the exact values.
  * All    -> "contextual chunk headers": prefix the embedded text with document + section so
              a chunk like "It grew 14%" still matches a query about the thing that grew.

LLM results are cached by content hash in output/enrich_cache.json, so re-ingesting is free.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, llm
from .models import Chunk

IMAGE_PROMPT = """You are indexing a figure from the PDF "{source}" (page {page}, section "{section}") \
for a search engine. Text printed near the figure:
<nearby_text>
{nearby}
</nearby_text>

Describe the figure so someone searching for its information would find it. Include:
1. What kind of figure it is (bar chart, photo, diagram, ...) and its subject.
2. Every readable title, axis label, legend entry, and number, with the values they map to.
3. The main takeaway(s) in one or two sentences.
Write plain prose, no preamble, under 200 words."""

TABLE_PROMPT = """You are indexing a table from the PDF "{source}" (page {page}, section "{section}") \
for a search engine. Text printed near the table:
<nearby_text>
{nearby}
</nearby_text>
<table>
{table}
</table>

Write a 2-4 sentence summary: what the table contains (rows, columns, units), and the most \
notable values, maxima/minima, or trends. Mention specific names and numbers. No preamble."""


def _load_cache() -> dict:
    if config.ENRICH_CACHE.exists():
        return json.loads(config.ENRICH_CACHE.read_text())
    return {}


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:24]


def _describe_image(c: Chunk) -> str:
    prompt = IMAGE_PROMPT.format(
        source=c.source, page=c.page_start, section=c.section, nearby=c.content or "(none)"
    )
    content = [llm.image_block(c.image_path), {"type": "text", "text": prompt}]
    return llm.text_of(llm.create(messages=[{"role": "user", "content": content}], effort="low"))


def _summarize_table(c: Chunk) -> str:
    prompt = TABLE_PROMPT.format(
        source=c.source, page=c.page_start, section=c.section,
        nearby=c.summary or "(none)", table=c.content,
    )
    return llm.text_of(llm.create(messages=[{"role": "user", "content": prompt}], effort="low"))


def _header(c: Chunk) -> str:
    pages = f"page {c.page_start}" if c.page_start == c.page_end else f"pages {c.page_start}-{c.page_end}"
    return f"Document: {c.source}\nSection: {c.section}\n({c.type}, {pages})\n\n"


def enrich_chunks(chunks: list[Chunk], pdf_path: Path, use_llm: bool = True) -> list[Chunk]:
    cache = _load_cache()

    def work(c: Chunk) -> None:
        if c.type == "image":
            image_bytes = (config.PROJECT_ROOT / c.image_path).read_bytes()
            k = _key("image", hashlib.sha256(image_bytes).hexdigest(), c.content)
            if use_llm and k not in cache:
                cache[k] = _describe_image(c)
            caption = c.content
            c.summary = cache.get(k, "")
            # What the agent reads as text for this image, alongside the image itself:
            c.content = "\n".join(x for x in [f"Caption/nearby text: {caption}" if caption else "",
                                              f"Description: {c.summary}" if c.summary else ""] if x)
            c.embed_text = _header(c) + c.content
        elif c.type == "table":
            k = _key("table", c.content, c.summary)
            if use_llm and k not in cache:
                cache[k] = _summarize_table(c)
            caption = c.summary
            c.summary = cache.get(k, caption)
            c.embed_text = _header(c) + (f"Caption: {caption}\n" if caption else "") + \
                f"Summary: {c.summary}\n\n{c.content}"
        else:
            c.embed_text = _header(c) + c.content

    # LLM calls are I/O bound; a few in parallel keeps ingestion quick.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, chunks))

    config.ENRICH_CACHE.write_text(json.dumps(cache, indent=2))
    out = config.doc_output_dir(pdf_path) / "chunks_enriched.json"
    out.write_text(json.dumps([c.to_dict() for c in chunks], indent=2))
    return chunks
