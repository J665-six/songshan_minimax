"""按实验块精确切分消融日志并渲染成 PNG。

不依赖复杂正则，直接按已知的 4 个实验标题定位切分点。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MONO = r"C:\Windows\Fonts\consola.ttf"
CJK = r"C:\Windows\Fonts\msyh.ttc"

BG = (24, 26, 31)
FG = (220, 223, 228)
DIM = (140, 146, 158)
GREEN = (126, 231, 135)
YELLOW = (255, 214, 102)
RED = (255, 130, 130)
BANNER_BG = (60, 66, 78)

SIZE = 17
LH = 26
PX = 26
PT = 62
PB = 26


def color_for(line: str):
    s = line.strip()
    if not s:
        return FG
    if s.startswith("===="):
        return BANNER_BG
    if s.startswith("[") and "正确性" not in s:
        return YELLOW
    if "FAIL" in s:
        return RED
    if "加速比" in s or "GB/s" in s:
        return GREEN
    if s.startswith("实验"):
        return (110, 217, 255)      # 标题行用青色
    if s.startswith("   ") or s.startswith("["):
        return FG
    return DIM


def render(lines, out: Path, title: str):
    mono = ImageFont.truetype(MONO, SIZE)
    cjk = ImageFont.truetype(CJK, SIZE)

    def width(s):
        return sum((cjk if ord(c) > 0x2E80 else mono).getlength(c) for c in s)

    max_w = max((width(l) for l in lines), default=600)
    W = int(min(max(max_w + PX * 2, 1000), 1900))
    H = PT + LH * len(lines) + PB

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 40], fill=(38, 41, 49))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([16 + i * 22, 14, 30 + i * 22, 28], fill=c)
    d.text((92, 11), title, font=cjk, fill=(190, 196, 208))

    y = PT - 14
    for line in lines:
        x = PX
        col = color_for(line)
        for ch in line:
            f = cjk if ord(ch) > 0x2E80 else mono
            d.text((x, y), ch, font=f, fill=col)
            x += f.getlength(ch)
        y += LH

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"[png] {out.name}  {W}x{H}  {out.stat().st_size // 1024} KB")


def main():
    src = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    raw = src.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    lines = [l for l in raw if not l.startswith("[auth]") and not l.startswith("[exit]")]

    # 找四个实验标题所在行
    marks = []
    for i, l in enumerate(lines):
        if "实验 1：RMSNorm" in l or "实验 2：Add-RMSNorm" in l:
            marks.append(i)

    # 环境段：开头到第一个标题之前的 banner 行
    env_end = marks[0] - 2 if marks else len(lines)
    env_block = lines[:env_end]
    env_block = [l for l in env_block if l.strip()]
    render(env_block, outdir / "01-环境与峰值带宽.png", "实验环境")

    names = ["02-RMSNorm-bf16", "03-AddRMSNorm-bf16",
             "04-RMSNorm-fp16", "05-AddRMSNorm-fp16"]
    for idx, start in enumerate(marks):
        end = marks[idx + 1] - 2 if idx + 1 < len(marks) else len(lines)
        block = [l for l in lines[start - 1:end] if l.strip()]
        # 汇总之前截断
        cut = [i for i, l in enumerate(block) if "汇总" in l]
        if cut:
            block = block[:cut[0]]
        render(block, outdir / f"{names[idx]}.png", names[idx].split("-", 1)[1].replace("-", " "))

    # 完整日志
    render([l for l in lines if l.strip()],
           outdir / "00-完整日志.png", "NineToothed 消融实验 — 完整输出")


if __name__ == "__main__":
    main()
