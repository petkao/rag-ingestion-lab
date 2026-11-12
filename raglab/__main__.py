"""Command line entry point:  uv run python -m raglab <command> ...

  ingest <pdf> [--no-llm]     parse -> chunk -> enrich -> embed/store, writes output/<doc>/*
  parse <pdf>                 stage 1 only (no API calls) - look at elements.json + images/
  search "<query>" [-k N] [--type table|image|text]   raw retriever results, no LLM
  ask "<question>"            one question to the RAG agent
  chat                        multi-turn conversation with the RAG agent
  eval [questions.json]       retrieval hit@k / MRR on known questions
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import config


def cmd_ingest(pdf: Path, use_llm: bool) -> None:
    from .chunk import build_chunks
    from .enrich import enrich_chunks
    from .parse import parse_pdf
    from .report import write_report
    from .store import store_chunks

    t0 = time.time()
    elements = parse_pdf(pdf)
    by_type = {t: sum(e.type == t for e in elements) for t in ("text", "table", "image")}
    print(f"[1/4] parse   {len(elements)} elements {by_type}")

    chunks = build_chunks(elements, pdf)
    by_type = {t: sum(c.type == t for c in chunks) for t in ("text", "table", "image")}
    print(f"[2/4] chunk   {len(chunks)} chunks {by_type}")

    enrich_chunks(chunks, pdf, use_llm=use_llm)
    print(f"[3/4] enrich  {'captions + table summaries via ' + config.CLAUDE_MODEL if use_llm else 'skipped (--no-llm)'}")

    n = store_chunks(chunks)
    print(f"[4/4] store   {n} vectors -> {config.CHROMA_DIR.relative_to(config.PROJECT_ROOT)}/{config.COLLECTION}")

    report = write_report(pdf)
    print(f"done in {time.time() - t0:.1f}s. Inspect: {report.relative_to(config.PROJECT_ROOT)}")


def cmd_parse(pdf: Path) -> None:
    from .parse import parse_pdf

    for e in parse_pdf(pdf):
        label = "HEADING" if e.is_heading else e.type
        preview = (e.text or e.image_path).replace("\n", " ")[:90]
        print(f"p{e.page:<3} {label:<8} {preview}")
    print(f"\nwrote {config.doc_output_dir(pdf).relative_to(config.PROJECT_ROOT)}/elements.json")


def cmd_search(query: str, k: int, types: list[str] | None) -> None:
    from .retrieve import search

    for h in search(query, k=k, types=types):
        print(f"\n{h.score:.3f}  {h.type.upper()}  [{h.citation}]  § {h.section}")
        if h.image_path:
            print(f"       image: {h.image_path}")
        print("       " + h.content[:400].replace("\n", "\n       "))


def main() -> None:
    p = argparse.ArgumentParser(prog="raglab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest"); s.add_argument("pdf", type=Path); s.add_argument("--no-llm", action="store_true")
    s = sub.add_parser("parse"); s.add_argument("pdf", type=Path)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)
    s.add_argument("--type", action="append", choices=["text", "table", "image"])
    s = sub.add_parser("ask"); s.add_argument("question"); s.add_argument("-q", "--quiet", action="store_true")
    sub.add_parser("chat")
    s = sub.add_parser("eval"); s.add_argument("questions", type=Path, nargs="?", default=config.DATA_DIR / "eval_questions.json")
    s.add_argument("-k", type=int, default=config.DEFAULT_TOP_K)

    a = p.parse_args()
    if a.cmd == "ingest":
        cmd_ingest(a.pdf, use_llm=not a.no_llm)
    elif a.cmd == "parse":
        cmd_parse(a.pdf)
    elif a.cmd == "search":
        cmd_search(a.query, a.k, a.type)
    elif a.cmd == "ask":
        from .agent import RagAgent
        print("\n" + RagAgent(verbose=not a.quiet).ask(a.question))
    elif a.cmd == "chat":
        from .agent import RagAgent
        agent = RagAgent()
        print("Ask about your PDFs (empty line to quit).")
        while q := input("\nyou> ").strip():
            print("\nclaude> " + agent.ask(q))
    elif a.cmd == "eval":
        from .evaluate import evaluate
        evaluate(a.questions, k=a.k)


if __name__ == "__main__":
    main()
