#!/usr/bin/env python3
"""Turns docs/guide.md into docs/guide.pdf (run inside WSL: python3 docs/build_guide.py).

Supports the small Markdown subset the guide uses: # / ## / ### headings,
paragraphs, - and 1. lists, ``` code blocks, | tables |, > callouts,
**bold**, `code`, and <!-- pagebreak -->.
Needs: python3-reportlab and the DejaVu fonts (both from apt).
"""

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "guide.md"
OUTPUT = HERE / "guide.pdf"
FONTS = Path("/usr/share/fonts/truetype/dejavu")

ACCENT = colors.HexColor("#4f7a00")    # dark green (NVIDIA-ish, but readable)
INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#5b636b")
CODE_BG = colors.HexColor("#f4f5f7")
RULE = colors.HexColor("#d5d9de")
CALLOUT_BG = colors.HexColor("#eef5e3")

# ---- fonts and styles -------------------------------------------------------

pdfmetrics.registerFont(TTFont("Sans", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont("Mono-Bold", str(FONTS / "DejaVuSansMono-Bold.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")

BODY = ParagraphStyle("body", fontName="Sans", fontSize=10, leading=14.5, textColor=INK, spaceAfter=6,
                      alignment=TA_LEFT)
STYLES = {
    "title": ParagraphStyle("title", parent=BODY, fontName="Sans-Bold", fontSize=30, leading=36,
                            textColor=ACCENT, spaceAfter=16),
    "subtitle": ParagraphStyle("subtitle", parent=BODY, fontSize=12.5, leading=18, textColor=MUTED),
    "h2": ParagraphStyle("h2", parent=BODY, fontName="Sans-Bold", fontSize=17, leading=22, textColor=ACCENT,
                         spaceBefore=6, spaceAfter=8),
    "h3": ParagraphStyle("h3", parent=BODY, fontName="Sans-Bold", fontSize=12.5, leading=17, spaceBefore=10,
                         spaceAfter=4),
    "body": BODY,
    "question": ParagraphStyle("question", parent=BODY, fontName="Sans-Bold", spaceBefore=6, spaceAfter=2),
    "bullet": ParagraphStyle("bullet", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=3),
    "subbullet": ParagraphStyle("subbullet", parent=BODY, leftIndent=32, bulletIndent=20, spaceAfter=2),
    "callout": ParagraphStyle("callout", parent=BODY, fontSize=10.5, leading=15.5, backColor=CALLOUT_BG,
                              borderPadding=(9, 10, 9, 10), borderColor=ACCENT, borderWidth=0,
                              leftIndent=10, rightIndent=10, spaceBefore=8, spaceAfter=14),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=10.6, textColor=INK, backColor=CODE_BG,
                           borderPadding=(6, 8, 6, 8), leftIndent=8, rightIndent=8, spaceBefore=4,
                           spaceAfter=12),
    "cell": ParagraphStyle("cell", parent=BODY, fontSize=8.8, leading=12, spaceAfter=0),
    "cellhead": ParagraphStyle("cellhead", parent=BODY, fontName="Sans-Bold", fontSize=8.8, leading=12,
                               spaceAfter=0, textColor=colors.white),
}

# ---- inline formatting --------------------------------------------------------

def inline(text: str) -> str:
    """Markdown inline markup -> reportlab's mini-HTML."""
    text = text.replace("\\|", "|")
    # Swap code spans for placeholders first, so ** or * inside code stays literal,
    # while **bold text with `code` inside** still works.
    spans = []

    def stash(m):
        spans.append('<font face="Mono" color="#8a3b12">' + _escape(m.group(1)) + "</font>")
        return f"\x00{len(spans) - 1}\x00"

    text = _emphasis(_escape(re.sub(r"`([^`]+)`", stash, text)))
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _emphasis(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)

# ---- block parsing --------------------------------------------------------------

def split_row(line: str) -> list:
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip() for c in cells]


def make_table(rows: list, width: float) -> Table:
    header, body = rows[0], rows[1:]
    ncols = len(header)
    # Column widths proportional to (capped) text length, so long text columns get more room.
    weights = []
    for c in range(ncols):
        longest = max(len(r[c]) if c < len(r) else 0 for r in rows)
        weights.append(min(max(longest, 8), 60))
    total = sum(weights)
    widths = [width * w / total for w in weights]
    data = [[Paragraph(inline(h), STYLES["cellhead"]) for h in header]]
    for r in body:
        r = r + [""] * (ncols - len(r))
        data.append([Paragraph(inline(c), STYLES["cell"]) for c in r[:ncols]])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8f9fa")))
    t.setStyle(TableStyle(style))
    return t


def parse(md: str, width: float) -> list:
    lines = md.splitlines()
    story, para = [], []
    title_done = False
    in_title_page = True
    i = 0

    def flush():
        if para:
            style = STYLES["subtitle"] if in_title_page and title_done else STYLES["body"]
            story.append(Paragraph(inline(" ".join(para)), style))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "<!-- pagebreak -->":
            flush()
            story.append(PageBreak())
            in_title_page = False
        elif stripped.startswith("```"):
            flush()
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].expandtabs(4))
                i += 1
            story.append(Preformatted("\n".join(code), STYLES["code"]))
        elif stripped.startswith("# "):
            flush()
            story.append(Spacer(1, 5 * cm))
            story.append(Paragraph(inline(stripped[2:]), STYLES["title"]))
            title_done = True
        elif stripped.startswith("## "):
            flush()
            story.append(Paragraph(inline(stripped[3:]), STYLES["h2"]))
        elif stripped.startswith("### "):
            flush()
            story.append(Paragraph(inline(stripped[4:]), STYLES["h3"]))
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = split_row(lines[i])
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in row):  # skip the |---|---| line
                    rows.append(row)
                i += 1
            story.append(make_table(rows, width))
            story.append(Spacer(1, 8))
            continue
        elif stripped.startswith("> "):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(quote)), STYLES["callout"]))
            continue
        elif re.match(r"^(- |\d+\. )", stripped):
            flush()
            nested = line.startswith("  ")  # an indented item belongs to the item above it
            m = re.match(r"^(- |(\d+)\. )(.*)", stripped)
            bullet = "•" if m.group(1) == "- " else m.group(2) + "."
            item = [m.group(3)]
            # continuation lines of the same item are indented (but are not items themselves)
            while (i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip()
                   and not re.match(r"^(- |\d+\. )", lines[i + 1].strip())):
                i += 1
                item.append(lines[i].strip())
            style = STYLES["subbullet"] if nested else STYLES["bullet"]
            story.append(Paragraph(inline(" ".join(item)), style, bulletText=bullet))
        elif re.fullmatch(r"\*\*.+\*\*", stripped):  # a line that is all bold: a question / label
            flush()
            story.append(Paragraph(inline(stripped), STYLES["question"]))
        elif not stripped:
            flush()
        else:
            para.append(stripped)
        i += 1
    flush()
    return story


def decorate(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, A4[1] - 1.2 * cm, "kc: Guide")
    canvas.drawRightString(A4[0] - doc.rightMargin, 1.2 * cm, str(doc.page))
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, A4[1] - 1.35 * cm, A4[0] - doc.rightMargin, A4[1] - 1.35 * cm)
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                            topMargin=2.0 * cm, bottomMargin=2.0 * cm, title="kc: Guide",
                            author="kc project")
    story = parse(SOURCE.read_text(encoding="utf-8"), doc.width)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    sys.exit(main())
