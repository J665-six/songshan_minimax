"""检查目标段落的 run 切分情况，为安全替换做准备。"""

from __future__ import annotations

import sys

from docx import Document

SRC = sys.argv[1]
doc = Document(SRC)

KEYS = [
    ("P059-ada", "候选为 1.636"),
    ("P076-mha", "3.730×"),
    ("P079-long", "216.578"),
    ("P085-gate", "1.609×"),
    ("P091-snake", "1.547×"),
    ("P097-wn", "0.1506"),
    ("P103-ablation", "0.2066"),
    ("P070-ref", "图 2-4"),
    ("P086-ref", "图 2-3"),
    ("P080-mem", "--mem=40G"),
]

for tag, key in KEYS:
    hit = None
    for i, p in enumerate(doc.paragraphs):
        if key in p.text:
            hit = (i, p)
            break
    if hit is None:
        print(f"\n[{tag}] 未找到 {key!r}")
        continue
    i, p = hit
    print(f"\n[{tag}] para#{i}  runs={len(p.runs)}  style={p.style.name}")
    for j, r in enumerate(p.runs):
        t = r.text
        if not t:
            continue
        mark = " <== 含目标" if key in t else ""
        print(f"   run[{j}] {t[:110]!r}{mark}")
