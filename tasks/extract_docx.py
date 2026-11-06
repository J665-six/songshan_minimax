"""提取 docx 的结构：按顺序列出段落文字、表格、以及内嵌图片的落点。

同时把 media 导出到临时目录，便于逐张核对。
"""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.oxml.ns import qn

SRC = Path(sys.argv[1])
OUTDIR = Path(sys.argv[2])
OUTDIR.mkdir(parents=True, exist_ok=True)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def para_has_image(p) -> int:
    """返回该段落内的图片数量。"""
    return len(p._element.findall(f".//{W}drawing"))


def para_image_rids(p) -> list[str]:
    rids = []
    for blip in p._element.findall(f".//{{http://schemas.openxmlformats.org/drawingml/2006/main}}blip"):
        rid = blip.get(f"{R}embed")
        if rid:
            rids.append(rid)
    return rids


def main() -> int:
    doc = Document(str(SRC))
    body = doc.element.body
    rels = doc.part.rels

    idx = 0
    img_no = 0
    lines = []

    for child in body.iterchildren():
        tag = child.tag
        if tag == f"{W}p":
            from docx.text.paragraph import Paragraph
            p = Paragraph(child, doc)
            text = p.text.strip()
            n_img = para_has_image(p)
            if n_img:
                for rid in para_image_rids(p):
                    img_no += 1
                    target = rels[rid].target_ref if rid in rels else "?"
                    lines.append(f"[图 {img_no:02d}] <-- rId={rid}  file={target}")
            if text:
                idx += 1
                style = p.style.name if p.style is not None else ""
                prefix = f"[P{idx:03d}]"
                if style and style != "Normal":
                    prefix += f" <{style}>"
                lines.append(f"{prefix} {text}")
        elif tag == f"{W}tbl":
            from docx.table import Table
            t = Table(child, doc)
            lines.append(f"[表格] {len(t.rows)} 行 x {len(t.columns)} 列")
            for row in t.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                lines.append("   | " + " | ".join(cells))

    text = "\n".join(lines)
    (OUTDIR / "structure.txt").write_text(text, encoding="utf-8")

    # 导出 media
    with ZipFile(str(SRC)) as z:
        media = [n for n in z.namelist() if n.startswith("word/media/")]
        for n in media:
            data = z.read(n)
            dst = OUTDIR / Path(n).name
            dst.write_bytes(data)

    print(f"段落/元素行数: {len(lines)}")
    print(f"图片数: {img_no}")
    print(f"media 文件数: {len(media)}")
    print(f"输出目录: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
