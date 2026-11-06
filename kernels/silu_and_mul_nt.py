"""NineToothed SiLU-and-Mul v3：二维 tile，与 rms_norm 相同的成功模式。

对照实验（seq=8192, bf16, 该机器）：
    torch copy  0.351 ms  1337 GB/s
    torch mul   0.529 ms  1331 GB/s   <- 三流元素级运算的真实上限
    torch silu  0.710 ms   662 GB/s   <- silu 本身受超越函数限制
    IC 基线     1.513 ms   465 GB/s
    NT v1/v2    1.540 ms   457 GB/s   <- 一维平铺，代码生成差

结论：目标不是打败 465，而是逼近 1331 的访存上限（silu 的现实目标是 ~660）。

用法：
    python kernels/silu_and_mul_nt.py --impl sigmoid --seqs 8192
"""

from __future__ import annotations

import argparse
import time

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

D = 14336
BLOCK_M = block_size(1, 4)


def arrangement(gate, up, y):
    return (
        gate.tile((BLOCK_M, D)),
        up.tile((BLOCK_M, D)),
        y.tile((BLOCK_M, D)),
    )


def application(gate, up, y):
    # tl.sigmoid 只接受 fp32，先做类型提升
    g = gate * 1.0
    y = g * ntl.sigmoid(g) * up  # noqa: F841


def application_mul(gate, up, y):
    """诊断用：纯乘法，衡量访存上限。"""
    y = gate * up  # noqa: F841


IMPLS = {"sigmoid": application, "mul": application_mul}


def build(impl: str, block_m: int | None = None, num_warps: int | None = None):
    global BLOCK_M
    if block_m is not None:
        BLOCK_M = block_size(block_m, block_m)
    tensors = (Tensor(2), Tensor(2), Tensor(2))
    return ninetoothed.make(arrangement, IMPLS[impl], tensors, num_warps=num_warps)


def reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)


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
    parser.add_argument("--impl", default="sigmoid", choices=list(IMPLS))
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--warps", type=int, default=None)
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    kernel = build(args.impl, args.block_m, args.warps)
    print(f"[kernel] built impl={args.impl} block_m={args.block_m} warps={args.warps}")

    for seq in [int(s) for s in args.seqs.split(",")]:
        gate = torch.randn(seq, D, device="cuda", dtype=dtype)
        up = torch.randn(seq, D, device="cuda", dtype=dtype)
        out = torch.empty(seq, D, device="cuda", dtype=dtype)

        kernel(gate, up, out)
        ref = reference(gate, up)
        torch.cuda.synchronize()

        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        diff = (out.float() - ref.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / ref.float().abs().clamp_min(1e-6)).max().item()

        t = bench(lambda: kernel(gate, up, out))
        gbs = (3 * seq * D * 2) / (t / 1000.0) / 1e9

        print(f"[nt] {args.dtype} seq={seq:<6} {t:8.4f} ms  {gbs:7.1f} GB/s  "
              f"cos={cos:.6f}  max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
