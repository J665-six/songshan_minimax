"""NineToothed SiLU-and-Mul v4: fixed row width + row tiling.

Key insight from add_rms_norm's success: when the tile's last dimension is a
compile-time constant, the mask is trivially true along the contiguous axis,
so Triton can vectorize bf16 loads to 128-bit. A 1D flat tiling keeps the mask
dynamic along the contiguous axis and collapses to scalar loads.

Implementation: reinterpret [M, 14336] contiguous memory as [M*7, 2048]
(14336 = 7 * 2048) and tile by (1, W). Pure view, no data movement.

--static additionally fixes both dimensions as compile-time constants to try
to remove the mask entirely.
"""

from __future__ import annotations

import argparse
import time

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

D = 14336
BLOCK_M = block_size(1, 2)


def make_kernel(W: int, rows: int | None = None, impl: str = "silu"):
    def arrangement(gate, up, y):
        return (
            gate.tile((BLOCK_M, W)),
            up.tile((BLOCK_M, W)),
            y.tile((BLOCK_M, W)),
        )

    def application(gate, up, y):
        g = gate * 1.0
        y = g * ntl.sigmoid(g) * up  # noqa: F841

    def application_copy(gate, up, y):
        y = gate  # noqa: F841

    body = application_copy if impl == "copy" else application
    if rows is None:
        tensors = (Tensor(2), Tensor(2), Tensor(2))
    else:
        shape = (rows, W)
        tensors = (
            Tensor(2, shape=shape),
            Tensor(2, shape=shape),
            Tensor(2, shape=shape),
        )
    return ninetoothed.make(arrangement, body, tensors)


def reference(gate, up):
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
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--impl", default="silu", choices=["silu", "copy"])
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    W = args.width
    assert D % W == 0, "width must divide 14336"

    rows = None
    if args.static:
        rows = int(args.seqs.split(",")[0]) * D // W
    kernel = make_kernel(W, rows, args.impl)
    print(f"[kernel] built width={W} static={args.static}")

    for seq in [int(s) for s in args.seqs.split(",")]:
        gate = torch.randn(seq, D, device="cuda", dtype=dtype)
        up = torch.randn(seq, D, device="cuda", dtype=dtype)
        out = torch.empty(seq, D, device="cuda", dtype=dtype)

        g2 = gate.view(-1, W)
        u2 = up.view(-1, W)
        o2 = out.view(-1, W)

        kernel(g2, u2, o2)
        ref = reference(gate, up)
        torch.cuda.synchronize()

        cos = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        diff = (out.float() - ref.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / ref.float().abs().clamp_min(1e-6)).max().item()

        t = bench(lambda: kernel(g2, u2, o2))
        gbs = (3 * seq * D * 2) / (t / 1000.0) / 1e9

        print(f"[nt] W={W} static={args.static} {args.dtype} seq={seq:<6} "
              f"{t:8.4f} ms  {gbs:7.1f} GB/s  cos={cos:.6f}  "
              f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
