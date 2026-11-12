"""Stage 4 - EMBED + STORE: vectors into ChromaDB, with metadata for filtering and citations.

Each row in the vector store holds:
  id         stable chunk id (re-ingesting the same PDF overwrites, never duplicates)
  embedding  OpenAI text-embedding-3-small vector of chunk.embed_text
  document   chunk.embed_text (what was embedded, handy for debugging)
  metadata   type, source, pages, section, image_path, content, summary
             -> filters like {"type": "table"} and page citations in answers

Chroma is used because it runs in-process with zero setup. The same rows map 1:1 onto
pgvector (ad-engine's store): one table with an `embedding vector(1536)` column + metadata columns.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from openai import OpenAI

from . import config
from .models import Chunk


@lru_cache
def _openai() -> OpenAI:
    return OpenAI()  # reads OPENAI_API_KEY


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), 100):  # batch to stay well under request limits
        resp = _openai().embeddings.create(model=config.EMBED_MODEL, input=texts[i : i + 100])
        vectors.extend(d.embedding for d in resp.data)
    return vectors


@lru_cache
def collection():
    db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # cosine distance: 0 = same direction, 2 = opposite. similarity = 1 - distance
    return db.get_or_create_collection(
        config.COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )


def list_documents() -> list[dict]:
    """One entry per ingested PDF, with chunk counts by type."""
    metas = collection().get(include=["metadatas"])["metadatas"]
    docs: dict[str, dict] = {}
    for m in metas:
        d = docs.setdefault(m["source"], {"source": m["source"], "text": 0, "table": 0, "image": 0,
                                          "pages": 0})
        d[m["type"]] += 1
        d["pages"] = max(d["pages"], m["page_end"])
    return sorted(docs.values(), key=lambda d: d["source"])


def delete_document(source: str) -> None:
    collection().delete(where={"source": source})


def store_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    col = collection()
    # Make re-ingestion idempotent: remove this document's old rows first
    col.delete(where={"source": chunks[0].source})
    vectors = embed([c.embed_text for c in chunks])
    col.add(
        ids=[c.id for c in chunks],
        embeddings=vectors,
        documents=[c.embed_text for c in chunks],
        metadatas=[
            {
                "type": c.type,
                "source": c.source,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": c.section,
                "image_path": c.image_path,
                "content": c.content,
                "summary": c.summary,
            }
            for c in chunks
        ],
    )
    return len(chunks)
