"""Generate data/sample_report.pdf: a fictional company report with text, tables, and images.

Some facts deliberately appear in only ONE modality, so you can test whether ingestion captured it:
  * only in a table : Q3 Pacific Northwest revenue, Ethiopia fair-trade share
  * only in an image: satisfaction score per store format (bar chart), roastery city (diagram)
  * only in text    : CEO name, 2026 store-opening target

Run:  uv run python scripts/make_sample_pdf.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = DATA / "_figures"


def make_bar_chart(path: Path) -> None:
    formats = ["Café", "Drive-thru", "Kiosk", "Online"]
    scores = [88, 71, 64, 79]
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    bars = ax.bar(formats, scores, color=["#7c3aed", "#0ea5e9", "#f59e0b", "#10b981"])
    ax.bar_label(bars, labels=[str(s) for s in scores], padding=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Satisfaction score (0-100)")
    ax.set_title("Customer satisfaction by store format, 2025")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_supply_chain_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.2, "Partner farms\nColombia · Ethiopia\n· Guatemala", "#fde68a"),
        (2.8, "Green-bean\nwarehouse\nOakland, CA", "#bfdbfe"),
        (5.4, "Central roastery\nBoise, Idaho\n(48h roast-to-ship)", "#fecaca"),
        (8.0, "212 stores\n+ online", "#bbf7d0"),
    ]
    for x, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.7), 1.8, 1.6, boxstyle="round,pad=0.05", fc=color, ec="#334155"))
        ax.text(x + 0.9, 1.5, label, ha="center", va="center", fontsize=8)
    for x in (2.0, 4.6, 7.2):
        ax.add_patch(FancyArrowPatch((x + 0.05, 1.5), (x + 0.75, 1.5), arrowstyle="-|>", mutation_scale=14))
    ax.set_title("Helios bean-to-cup supply chain", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def styled_table(rows, col_widths):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    return t


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    chart, diagram = FIG / "satisfaction.png", FIG / "supply_chain.png"
    make_bar_chart(chart)
    make_supply_chain_diagram(diagram)

    ss = getSampleStyleSheet()
    H1, H2, P = ss["Title"], ss["Heading2"], ss["BodyText"]
    cap = ss["Italic"]
    story = [
        Paragraph("Helios Coffee Roasters — 2025 Annual Report", H1),
        Paragraph("Executive Summary", H2),
        Paragraph(
            "Helios Coffee Roasters closed 2025 with total revenue of $184.6 million, up 12% year over "
            "year, driven by strong café traffic and the launch of our subscription program. Chief "
            "Executive Officer Mariana Okafor, who joined in March 2024, led a reorganisation of the "
            "store operations team into four regional units. Operating margin improved from 9.1% to "
            "10.4% as we renegotiated dairy contracts and reduced waste.", P),
        Paragraph(
            "Our subscription program, Helios Daily, reached 61,000 active members by December. "
            "Members visit 3.2 times per week on average, compared with 1.4 visits for non-members, "
            "and account for 27% of café revenue. We plan to extend member pricing to online bean "
            "orders in the first half of 2026.", P),
        Paragraph("Strategy for 2026", H2),
        Paragraph(
            "In 2026 we will open 30 new stores, focusing on the Mountain region where our brand "
            "awareness is lowest. We will also pilot a reusable-cup deposit scheme in 15 Seattle "
            "cafés, invest $6 million in roastery automation, and introduce a decaf single-origin line. "
            "Capital expenditure is budgeted at $22 million, funded entirely from operating cash flow.", P),
        PageBreak(),
        Paragraph("Regional Performance", H2),
        Paragraph(
            "Revenue grew in every region. The Pacific Northwest remains our largest market, while the "
            "Southwest posted the fastest growth after 18 openings in Phoenix and Tucson. Quarterly "
            "revenue by region is shown in Table 1 (US$ millions).", P),
        Spacer(1, 6),
        Paragraph("Table 1: Quarterly revenue by region, 2025 (US$ millions)", cap),
        styled_table([
            ["Region", "Q1", "Q2", "Q3", "Q4", "Full year"],
            ["Pacific Northwest", "17.2", "18.9", "19.6", "20.3", "76.0"],
            ["California", "10.1", "11.0", "11.8", "12.4", "45.3"],
            ["Southwest", "6.4", "7.9", "8.8", "9.6", "32.7"],
            ["Mountain", "6.9", "7.4", "7.9", "8.4", "30.6"],
            ["Total", "40.6", "45.2", "48.1", "50.7", "184.6"],
        ], [1.8 * inch] + [0.8 * inch] * 5),
        Spacer(1, 10),
        Paragraph(
            "Store-level economics improved in all formats. New stores reached break-even after "
            "an average of 7 months, down from 9 months in 2024.", P),
        Paragraph("Customer Experience", H2),
        Paragraph(
            "We survey 5,000 customers every quarter. Satisfaction by store format is shown in Figure 1. "
            "The gap between formats tells us where to invest in 2026.", P),
        Image(str(chart), width=5.4 * inch, height=3.06 * inch),
        Paragraph("Figure 1: Customer satisfaction by store format, 2025 survey", cap),
        PageBreak(),
        Paragraph("Supply Chain and Sustainability", H2),
        Paragraph(
            "All beans are roasted in a single central facility and shipped to stores within 48 hours. "
            "Figure 2 shows how coffee moves from partner farms to our stores.", P),
        Image(str(diagram), width=6.3 * inch, height=2.16 * inch),
        Paragraph("Figure 2: Bean-to-cup supply chain", cap),
        Spacer(1, 8),
        Paragraph(
            "We buy directly from 140 partner farms. Table 2 summarises sourcing by origin country.", P),
        Paragraph("Table 2: Green coffee sourcing by origin, 2025", cap),
        styled_table([
            ["Origin", "Tonnes", "Share of volume", "Fair-trade certified"],
            ["Colombia", "1,850", "41%", "78%"],
            ["Ethiopia", "1,120", "25%", "92%"],
            ["Guatemala", "860", "19%", "65%"],
            ["Other", "670", "15%", "40%"],
        ], [1.5 * inch, 1.0 * inch, 1.4 * inch, 1.6 * inch]),
        Spacer(1, 8),
        Paragraph(
            "Our goal is 85% fair-trade certified volume by 2027. Roastery emissions fell 18% after we "
            "switched to a renewable-electricity contract in May. Packaging is now 94% recyclable; the "
            "remaining 6% is the valve on our one-kilogram bags, which we expect to replace in 2026.", P),
    ]
    out = DATA / "sample_report.pdf"
    SimpleDocTemplate(str(out), pagesize=letter, title="Helios 2025 Annual Report").build(story)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
