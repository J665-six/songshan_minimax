"""NineToothed RMSNorm（MiniMax-H3, hidden=5376）

按官方 README 的 arrange-and-apply 范式：
    kernel = ninetoothed.make(arrangement, application, tensors)

运行时传 torch 张量即可。
"""

from __future__ import annotations

import argparse
import time

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

N = 5376
EPS = 1e-5

BLOCK_M = block_size(1, 8)


def arrangement(x, w, y):
    """每个 program 处理 BLOCK_M 行，tile 覆盖整行（N 固定为常量）。"""
    y_arranged = y.tile((BLOCK_M, N))
    x_arranged = x.tile((BLOCK_M, N))
    w_arranged = w.tile((N,)).expand((x_arranged.shape[0], -1))
    return x_arranged, w_arranged, y_arranged


def application(x, w, y):
    # 用 x.shape[1]（tile 宽度，编译期符号）而不是 Python 全局 N，
    # SSA 后端不认识模块级常量。
    mean_square = ntl.sum(x * x, axis=1) / x.shape[1]
    inv_rms = 1.0 / ntl.sqrt(mean_square + 1e-5)
    y = x * inv_rms[:, None] * w  # noqa: F841


def build(block_m: int | None = None, num_warps: int | None = None):
    global BLOCK_M
    if block_m is not None:
        BLOCK_M = block_size(block_m, block_m)
    tensors = (Tensor(2), Tensor(1), Tensor(2))
    return ninetoothed.make(
        arrangement, application, tensors, num_warps=num_warps
    )


def reference(x: torch.Tensor, w: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf / torch.sqrt(ms + eps) * w.float()).to(x.dtype)


def bench(fn, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", default="2048,8192")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--sweep", action="store_true", help="扫描 BLOCK_M x num_warps")
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.sweep:
        x = torch.randn(8192, N, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(x)
        ref = reference(x, w)
        best = None
        for bm in (1, 2, 4, 8):
            for warps in (2, 4, 8):
                try:
                    kernel = build(bm, warps)
                    kernel(x, w, out)
                    torch.cuda.synchronize()
                    cos = torch.nn.functional.cosine_similarity(
                        out.float().flatten(), ref.float().flatten(), dim=0
                    ).item()
                    t = bench(lambda: kernel(x, w, out))
                    gbs = (2 * x.shape[0] * N * 2) / (t / 1000.0) / 1e9
                    print(f"[sweep] BLOCK_M={bm} warps={warps}  {t:7.4f} ms  "
                          f"{gbs:7.1f} GB/s  cos={cos:.6f}")
                    if best is None or t < best[0]:
                        best = (t, bm, warps, gbs)
                except Exception as exc:  # noqa: BLE001
                    print(f"[sweep] BLOCK_M={bm} warps={warps}  FAILED: {exc}")
        if best:
            print(f"[best] BLOCK_M={best[1]} warps={best[2]}  {best[0]:.4f} ms  "
                  f"{best[3]:.1f} GB/s")
        return 0

    kernel = build()
    print("[kernel] built")

    for seq in [int(s) for s in args.seqs.split(",")]:
        x = torch.randn(seq, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype)
        out = torch.empty_like(x)

        kernel(x, w, out)
        ref = reference(x, w)
        torch.cuda.synchronize()

        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        diff = (out.float() - ref.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / ref.float().abs().clamp_min(1e-6)).max().item()

        t = bench(lambda: kernel(x, w, out))
        gbs = (2 * seq * N * 2) / (t / 1000.0) / 1e9

        print(f"[nt] {args.dtype} seq={seq:<6} {t:8.4f} ms  {gbs:7.1f} GB/s  "
              f"cos={cos:.6f}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
