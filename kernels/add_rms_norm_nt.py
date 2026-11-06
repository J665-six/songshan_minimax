"""NineToothed 融合 Add + RMSNorm。

MiniMax-H3 的每个 DiT block 里都有残差相加后接 RMSNorm 的模式：
    residual = residual + x
    y = rms_norm(residual, w)
InfiniCore 已有 add_rms_norm 算子，但只跑到 605 GB/s（峰值 1342 的 45%）。

这里用 rms_norm 已验证有效的行归约模式重写，一次读完、一次写归一化结果、
一次写残差，避免两次 kernel 往返。
"""

from __future__ import annotations

import argparse
import time

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

N = 5376
BLOCK_M = block_size(1, 4)


def arrangement(a, b, w, y, res):
    return (
        a.tile((BLOCK_M, N)),
        b.tile((BLOCK_M, N)),
        w.tile((N,)).expand((a.shape[0], -1)),
        y.tile((BLOCK_M, N)),
        res.tile((BLOCK_M, N)),
    )


def application(a, b, w, y, res):
    s = a + b
    mean_square = ntl.sum(s * s, axis=1) / s.shape[1]
    inv_rms = 1.0 / ntl.sqrt(mean_square + 1e-5)
    y = s * inv_rms[:, None] * w  # noqa: F841
    res = s  # noqa: F841


def build(block_m: int | None = None, num_warps: int | None = None):
    global BLOCK_M
    if block_m is not None:
        BLOCK_M = block_size(block_m, block_m)
    tensors = (Tensor(2), Tensor(2), Tensor(1), Tensor(2), Tensor(2))
    return ninetoothed.make(arrangement, application, tensors, num_warps=num_warps)


def reference(a, b, w, eps=1e-5):
    s = (a.float() + b.float())
    ms = s.pow(2).mean(dim=-1, keepdim=True)
    y = (s / torch.sqrt(ms + eps) * w.float()).to(a.dtype)
    return y, s.to(a.dtype)


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
    parser.add_argument("--seqs", default="8192")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--warps", type=int, default=None)
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    kernel = build(args.block_m, args.warps)
    print(f"[kernel] built block_m={args.block_m} warps={args.warps}")

    for seq in [int(s) for s in args.seqs.split(",")]:
        a = torch.randn(seq, N, device="cuda", dtype=dtype)
        b = torch.randn(seq, N, device="cuda", dtype=dtype)
        w = torch.randn(N, device="cuda", dtype=dtype)
        y = torch.empty(seq, N, device="cuda", dtype=dtype)
        res = torch.empty(seq, N, device="cuda", dtype=dtype)

        kernel(a, b, w, y, res)
        ref_y, ref_res = reference(a, b, w)
        torch.cuda.synchronize()

        cos = torch.nn.functional.cosine_similarity(
            y.float().flatten(), ref_y.float().flatten(), dim=0
        ).item()
        cos_r = torch.nn.functional.cosine_similarity(
            res.float().flatten(), ref_res.float().flatten(), dim=0
        ).item()
        diff = (y.float() - ref_y.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / ref_y.float().abs().clamp_min(1e-6)).max().item()

        t = bench(lambda: kernel(a, b, w, y, res))
        # 读 a+b，写 y + res
        gbs = (4 * seq * N * 2) / (t / 1000.0) / 1e9

        print(f"[nt] {args.dtype} seq={seq:<6} {t:8.4f} ms  {gbs:7.1f} GB/s  "
              f"cos={cos:.6f} cos_res={cos_r:.6f}  max_abs={max_abs:.3e}  "
              f"max_rel={max_rel:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
