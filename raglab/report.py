"""Write output/<doc>/report.html: every chunk, exactly as it was embedded, with images inline.
Open it in a browser to *see* what your retriever can and can't find."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

from . import config

COLORS = {"text": "#2563eb", "table": "#16a34a", "image": "#c2410c"}


def write_report(pdf_path: Path) -> Path:
    out_dir = config.doc_output_dir(pdf_path)
    chunks = json.loads((out_dir / "chunks_enriched.json").read_text())
    counts = {t: sum(c["type"] == t for c in chunks) for t in COLORS}

    cards = []
    for c in chunks:
        img = ""
        if c["image_path"]:
            rel = os.path.relpath(config.PROJECT_ROOT / c["image_path"], out_dir)
            img = f'<img src="{html.escape(rel)}" alt="">'
        pages = c["page_start"] if c["page_start"] == c["page_end"] else f'{c["page_start"]}-{c["page_end"]}'
        cards.append(f"""
<section class="card">
  <div class="meta"><span class="tag" style="background:{COLORS[c['type']]}">{c['type']}</span>
    p.{pages} · {html.escape(c['section'])} · <code>{html.escape(c['id'])}</code></div>
  {img}
  <h4>Embedded text ({len(c['embed_text'])} chars)</h4>
  <pre>{html.escape(c['embed_text'])}</pre>
</section>""")

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ingestion report: {html.escape(pdf_path.name)}</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px;background:#f8fafc;color:#0f172a}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;margin:12px 0}}
 .meta{{font-size:13px;color:#475569}} .tag{{color:#fff;border-radius:4px;padding:1px 6px;margin-right:6px}}
 pre{{white-space:pre-wrap;background:#f1f5f9;padding:8px;border-radius:6px;font-size:13px}}
 img{{max-width:100%;max-height:360px;display:block;margin:8px 0;border:1px solid #e2e8f0}}
 h4{{margin:8px 0 4px;font-size:13px;color:#475569}}
</style></head><body>
<h1>{html.escape(pdf_path.name)}</h1>
<p>{len(chunks)} chunks: {counts['text']} text · {counts['table']} table · {counts['image']} image</p>
{''.join(cards)}
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(page)
    return path
