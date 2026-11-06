"""对报告做 run 级精确替换，保持原有格式不变。

只改数字、图号引用、--mem 参数，不动图片和样式。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

# (定位用唯一片段, 旧串, 新串)
EDITS: list[tuple[str, str, str]] = [
    # ---- ① AdaLN ----
    ("候选为 1.636 ms", "候选为 1.636 ms，加速 1.321×", "候选为 1.642 ms，加速 1.316×"),

    # ---- ② MHA 同进程 A/B ----
    (
        "3.730×",
        "1024 token 时由 1.153 ms 降至 0.309 ms，加速 3.730×，cosine 为 0.9999804",
        "1024 token 时由 1.152 ms 降至 0.310 ms，加速 3.715×，cosine 为 0.9999807",
    ),
    (
        "7.811×",
        "4096 token 时由 21.852 ms 降至 2.798 ms，加速 7.811×，cosine 为 0.9999800",
        "4096 token 时由 21.833 ms 降至 2.791 ms，加速 7.824×，cosine 为 0.9999799",
    ),

    # ---- ③ MHA 长序列 ----
    (
        "216.578",
        "耗时分别为 0.297、2.786、10.001、216.578 ms",
        "耗时分别为 0.298、2.782、9.999、216.676 ms",
    ),

    # ---- ④ 门控残差 ----
    (
        "1.609×",
        "8192 token 时，基线为 1.370 ms，候选为 0.852 ms，加速 1.609×；32768 token 时，基线为 5.446 ms，候选为 3.375 ms，加速 1.614×",
        "8192 token 时，基线为 1.370 ms，候选为 0.854 ms，加速 1.602×；32768 token 时，基线为 5.446 ms，候选为 3.372 ms，加速 1.615×",
    ),

    # ---- ⑤ SnakeBeta ----
    (
        "1.547×",
        "JIT 在长度 207、2048、16384、331200 时分别达到 1.547×、1.398×、2.404×、2.351×",
        "JIT 在长度 207、2048、16384、331200 时分别达到 1.681×、1.514×、2.438×、2.347×",
    ),
    (
        "2.007×",
        "AOT 在 16384 与 331200 时达到 1.121× 与 2.007×",
        "AOT 在 16384 与 331200 时达到 1.116× 与 2.010×",
    ),

    # ---- ⑥ weight-norm ----
    (
        "0.1506",
        "测试 Conv1d 由 0.1506 ms 降至 0.1136 ms，达到 1.326×",
        "测试 Conv1d 由 0.1547 ms 降至 0.1150 ms，达到 1.345×",
    ),

    # ---- ⑦ 离线候选 ----
    (
        "0.2066",
        "RMSNorm 由 0.4171 ms 降至 0.2066 ms，达到 2.02×",
        "RMSNorm 由 0.4172 ms 降至 0.2050 ms，达到 2.04×",
    ),
    (
        "0.5694",
        "Add-RMSNorm 由 0.5694 ms 降至 0.3167 ms，达到 1.80×",
        "Add-RMSNorm 由 0.5685 ms 降至 0.3163 ms，达到 1.80×",
    ),
    (
        "1.52×",
        "Add-RMSNorm 达到 1.52×",
        "Add-RMSNorm 达到 1.54×",
    ),
    (
        "0.26%",
        "3 轮波动最大约 0.26%",
        "3 轮波动最大约 2.3%（fp16 RMSNorm 第三轮为离群值，取中位）",
    ),

    # ---- ⑧ 图号引用 ----
    (
        "图 2-4",
        "性能证据与 Q/K RMSNorm 共用图 2-4。",
        "性能证据见本节截图。",
    ),
    (
        "图 2-3",
        "门控残差的性能与正确性结果包含在图 2-3 中。",
        "门控残差的性能与正确性结果见本节截图。",
    ),

    # ---- ⑨ --mem ----
    (
        "--mem=40G",
        "--gres=dcu:1 --mem=40G --time=00:15:00",
        "--gres=dcu:1 --mem=30G --time=00:15:00",
    ),
]


def main() -> int:
    doc = Document(str(SRC))
    done, missed = [], []

    for key, old, new in EDITS:
        found = False
        for p in doc.paragraphs:
            if key not in p.text:
                continue
            for r in p.runs:
                if old in r.text:
                    r.text = r.text.replace(old, new)
                    found = True
                    break
            if not found:
                # 跨 run 情况：合并到第一个 run
                joined = "".join(r.text for r in p.runs)
                if old in joined:
                    base = p.runs[0]
                    for r in p.runs[1:]:
                        r.text = ""
                    base.text = joined.replace(old, new)
                    found = True
            if found:
                break
        (done if found else missed).append(key)

    doc.save(str(DST))

    print(f"成功替换: {len(done)}/{len(EDITS)}")
    for k in missed:
        print(f"  [未命中] {k}")

    # 校验
    chk = Document(str(DST))
    all_text = "\n".join(p.text for p in chk.paragraphs)
    for bad in ("1.636", "3.730×", "7.811×", "216.578", "1.609×",
                "1.547×", "2.007×", "0.1506", "0.2066", "0.5694",
                "1.52×", "0.26%", "图 2-3", "图 2-4", "--mem=40G"):
        if bad in all_text:
            print(f"  [残留] {bad}")
    print("校验完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
