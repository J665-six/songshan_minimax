"""把技术报告 Markdown 转成排版好的 DOCX。

支持：标题层级、段落、表格、代码块、粗体、行内代码。
中文用 宋体/微软雅黑，英文用 Times New Roman。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

BODY_CN = "宋体"
HEAD_CN = "微软雅黑"
LATIN = "Times New Roman"
CODE_FONT = "Consolas"


def set_run_font(run, *, cn=BODY_CN, latin=LATIN, size=None, bold=None, color=None):
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), cn)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def shade(paragraph, hex_fill="F2F2F2"):
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    ppr.append(shd)


def add_inline(paragraph, text, *, size=10.5, cn=BODY_CN):
    """处理 **粗体** 与 `代码`。"""
    tokens = re.split(r"(\*\*.+?\*\*|`[^`]+`)", text)
    for token in tokens:
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, cn=cn, size=size, bold=True)
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, cn=cn, latin=CODE_FONT, size=size - 0.5)
            run.font.color.rgb = RGBColor(0xC7, 0x25, 0x4E)
        else:
            run = paragraph.add_run(token)
            set_run_font(run, cn=cn, size=size)


def add_table(document, rows):
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, text in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_inline(para, text, size=9.5, cn=HEAD_CN)
        for run in para.runs:
            run.font.bold = True

    for row in body:
        cells = table.add_row().cells
        for i, text in enumerate(row[: len(header)]):
            cells[i].text = ""
            para = cells[i].paragraphs[0]
            add_inline(para, text, size=9.5)

    document.add_paragraph()


def build():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = Document()

    section = doc.sections[0]
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)

    style = doc.styles["Normal"]
    style.font.name = LATIN
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CN)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            for code_line in block:
                para = doc.add_paragraph()
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.space_before = Pt(0)
                para.paragraph_format.left_indent = Cm(0.5)
                shade(para)
                run = para.add_run(code_line if code_line else " ")
                set_run_font(run, latin=CODE_FONT, size=8.5)
            doc.add_paragraph()
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and re.match(
            r"^\|[\s:\-|]+\|$", lines[i + 1].strip()
        ):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            add_table(doc, rows)
            continue

        if stripped.startswith("# "):
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(stripped[2:])
            set_run_font(run, cn=HEAD_CN, size=18, bold=True)
            para.paragraph_format.space_after = Pt(6)
        elif stripped.startswith("## "):
            para = doc.add_paragraph()
            run = para.add_run(stripped[3:])
            set_run_font(run, cn=HEAD_CN, size=14, bold=True)
            para.paragraph_format.space_before = Pt(14)
            para.paragraph_format.space_after = Pt(6)
        elif stripped.startswith("### "):
            para = doc.add_paragraph()
            run = para.add_run(stripped[4:])
            set_run_font(run, cn=HEAD_CN, size=12, bold=True)
            para.paragraph_format.space_before = Pt(10)
            para.paragraph_format.space_after = Pt(4)
        elif stripped == "---":
            pass
        elif stripped.startswith("- "):
            para = doc.add_paragraph(style="List Bullet")
            add_inline(para, stripped[2:])
        elif re.match(r"^\d+\.\s", stripped):
            para = doc.add_paragraph()
            add_inline(para, stripped)
            para.paragraph_format.left_indent = Cm(0.6)
        elif stripped == "":
            pass
        else:
            para = doc.add_paragraph()
            para.paragraph_format.space_after = Pt(4)
            para.paragraph_format.line_spacing = 1.25
            add_inline(para, stripped)

        i += 1

    doc.save(DST)
    return DST


if __name__ == "__main__":
    out = build()
    print("saved:", out, out.stat().st_size, "bytes")
