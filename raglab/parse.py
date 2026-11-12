"""Stage 1 - PARSE: turn a PDF into typed Elements (text blocks, tables, images).

A PDF has no notion of "paragraph" or "table": it is drawing instructions that place glyphs,
lines and pictures at coordinates. Parsing means recovering structure from layout:

  * Text   -> PyMuPDF groups glyphs into blocks/lines/spans; span font size tells us headings.
  * Tables -> page.find_tables() detects grids from ruling lines and text alignment.
              Text inside a table's bbox is dropped from the text stream so it isn't indexed twice.
  * Images -> embedded raster images are pulled out by their xref and saved as PNG.
              Charts drawn as *vector* graphics are NOT images in the PDF; see the README exercise.

Output: output/<doc>/elements.json + output/<doc>/images/*.png
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pymupdf

from . import config
from .models import Element


def _overlaps_any(bbox, rects, min_ratio=0.5) -> bool:
    """True if `bbox` lies mostly inside any rect (used to drop text that belongs to a table)."""
    r = pymupdf.Rect(bbox)
    if r.is_empty:
        return False
    for other in rects:
        inter = r & other
        if not inter.is_empty and inter.get_area() / r.get_area() >= min_ratio:
            return True
    return False


def _body_font_size(doc: pymupdf.Document) -> float:
    """The most common span size in the document = body text. Headings are larger."""
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(round(span["size"], 1))
    return statistics.mode(sizes) if sizes else 10.0


def _text_near(rect: pymupdf.Rect, text_blocks: list[Element], max_gap=40) -> str:
    """Text blocks directly above or below a rect (captions like 'Figure 2: ...' live here)."""
    above, below = [], []
    for el in text_blocks:
        if el.is_heading:
            continue
        b = pymupdf.Rect(el.bbox)
        if not (b.x1 > rect.x0 and b.x0 < rect.x1):  # must overlap horizontally
            continue
        gap_above = rect.y0 - b.y1
        gap_below = b.y0 - rect.y1
        if -8 <= gap_above <= max_gap:   # small negative tolerance: boxes often touch/overlap
            above.append((gap_above, el.text))
        if -8 <= gap_below <= max_gap:
            below.append((gap_below, el.text))
    # only the closest block on each side, otherwise whole paragraphs get glued on
    picked = [min(side)[1] for side in (above, below) if side]
    return "\n".join(picked)


def _table_to_markdown(table, words) -> str:
    """Rebuild each cell from the words whose centre falls inside it.

    PyMuPDF's own table.extract()/to_markdown() can mangle cells (dropped spaces, right-aligned
    numbers split across lines). Assigning whole words by their centre point is more robust and
    a good example of why you should always eyeball parser output before trusting it.
    """
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell is None:  # merged cell
                cells.append("")
                continue
            r = pymupdf.Rect(cell)
            inside = [w[4] for w in words if r.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))]
            cells.append(" ".join(inside).replace("|", "\\|"))
        rows.append(cells)
    if not rows or not any(any(c for c in r) for r in rows):
        return ""
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _save_image(doc, xref: int, smask: int, path: Path) -> tuple[int, int]:
    """Normalise any embedded image (JPEG, CMYK, with alpha mask...) to an RGB(A) PNG."""
    pix = pymupdf.Pixmap(doc, xref)
    if smask:  # separate transparency mask
        pix = pymupdf.Pixmap(pix, pymupdf.Pixmap(doc, smask))
    if pix.n - pix.alpha >= 4:  # CMYK -> RGB
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    pix.save(path)
    return pix.width, pix.height


def parse_pdf(pdf_path: Path) -> list[Element]:
    out_dir = config.doc_output_dir(pdf_path)
    doc = pymupdf.open(pdf_path)
    body_size = _body_font_size(doc)
    elements: list[Element] = []
    seen_xrefs: set[int] = set()  # logos repeated on every page are the same xref; keep one

    for page in doc:
        pno = page.number + 1

        # ---- Tables first, so we know which text to exclude --------------------------
        table_rects = []
        tables = page.find_tables()
        words = page.get_text("words")
        page_tables = []
        for ti, table in enumerate(tables.tables):
            md = _table_to_markdown(table, words)
            if not md:
                continue
            rect = pymupdf.Rect(table.bbox)
            table_rects.append(rect)
            page_tables.append(
                Element(
                    id=f"p{pno}-table{ti}",
                    type="table",
                    page=pno,
                    bbox=list(rect),
                    text=md,
                    extra={"rows": table.row_count, "cols": table.col_count},
                )
            )

        # ---- Text blocks (outside tables) --------------------------------------------
        page_text = []
        for bi, block in enumerate(page.get_text("dict", sort=True)["blocks"]):
            if block["type"] != 0:  # 1 = image block; images are handled below
                continue
            spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "\n".join(
                "".join(s["text"] for s in line["spans"]).strip() for line in block["lines"]
            ).strip()
            if not text or _overlaps_any(block["bbox"], table_rects):
                continue
            max_size = max(s["size"] for s in spans)
            is_bold = all(s["flags"] & 16 for s in spans)  # bit 4 = bold
            is_heading = len(text) < 120 and (
                max_size >= body_size * config.HEADING_SIZE_RATIO or (is_bold and len(text) < 80)
            )
            page_text.append(
                Element(
                    id=f"p{pno}-text{bi}",
                    type="text",
                    page=pno,
                    bbox=list(block["bbox"]),
                    text=text,
                    is_heading=is_heading,
                    extra={"font_size": round(max_size, 1)},
                )
            )

        # ---- Images ------------------------------------------------------------------
        page_images = []
        for ii, img in enumerate(page.get_images(full=True)):
            xref, smask, w, h = img[0], img[1], img[2], img[3]
            if xref in seen_xrefs or w < config.MIN_IMAGE_PX or h < config.MIN_IMAGE_PX:
                continue
            seen_xrefs.add(xref)
            rects = page.get_image_rects(xref)
            rect = rects[0] if rects else page.rect
            path = out_dir / "images" / f"p{pno}-img{ii}.png"
            width, height = _save_image(doc, xref, smask, path)
            page_images.append(
                Element(
                    id=f"p{pno}-img{ii}",
                    type="image",
                    page=pno,
                    bbox=list(rect),
                    image_path=str(path.relative_to(config.PROJECT_ROOT)),
                    nearby_text=_text_near(rect, page_text),
                    extra={"width": width, "height": height},
                )
            )

        for t in page_tables:
            t.nearby_text = _text_near(pymupdf.Rect(t.bbox), page_text)

        # Keep reading order on the page: sort everything top-to-bottom, then left-to-right.
        page_elements = page_text + page_tables + page_images
        page_elements.sort(key=lambda e: (round(e.bbox[1]), e.bbox[0]))
        elements.extend(page_elements)

    (out_dir / "elements.json").write_text(json.dumps([e.to_dict() for e in elements], indent=2))
    return elements
