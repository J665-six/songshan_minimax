"""第二遍：修补结果分析段落里残留的旧数字。"""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

EDITS = [
    (
        "提高到 4096",
        "加速比由 1024 token 的 3.730× 提高到 4096 token 的 7.811×",
        "加速比由 1024 token 的 3.715× 提高到 4096 token 的 7.824×",
    ),
    (
        "长度达到 2.007×",
        "AOT 在 331200 长度达到 2.007×",
        "AOT 在 331200 长度达到 2.010×",
    ),
]

doc = Document(str(SRC))
for key, old, new in EDITS:
    hit = False
    for p in doc.paragraphs:
        if key not in p.text:
            continue
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new)
                hit = True
                break
        if not hit:
            joined = "".join(r.text for r in p.runs)
            if old in joined:
                base = p.runs[0]
                for r in p.runs[1:]:
                    r.text = ""
                base.text = joined.replace(old, new)
                hit = True
        if hit:
            print(f"[ok] {key}")
            break
    if not hit:
        print(f"[miss] {key}")

doc.save(str(DST))

# 终检
chk = Document(str(DST))
text = "\n".join(p.text for p in chk.paragraphs)
STALE = ["1.636", "1.321×", "3.730×", "7.811×", "0.309", "2.798",
         "216.578", "2.786", "10.001", "1.609×", "1.547×", "1.398×",
         "2.404×", "2.007×", "0.1506", "0.1136", "1.326×",
         "0.2066", "0.4171", "0.5694", "0.3167", "1.52×", "0.26%",
         "图 2-3", "图 2-4", "--mem=40G"]
left = [s for s in STALE if s in text]
print("残留:", left if left else "无")
