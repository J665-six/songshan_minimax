"""把 NineToothed RMSNorm 导出成 C++/CUDA 源码，供 InfiniCore 的 hygon 后端调用。

运行后输出目录里会有生成的 kernel 源码与启动器。
"""

from __future__ import annotations

import argparse
import os

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

N = 5376
BLOCK_M = block_size(1, 1)


def arrangement(x, w, y):
    y_arranged = y.tile((BLOCK_M, N))
    x_arranged = x.tile((BLOCK_M, N))
    w_arranged = w.tile((N,)).expand((x_arranged.shape[0], -1))
    return x_arranged, w_arranged, y_arranged


def application(x, w, y):
    mean_square = ntl.sum(x * x, axis=1) / x.shape[1]
    inv_rms = 1.0 / ntl.sqrt(mean_square + 1e-5)
    y = x * inv_rms[:, None] * w  # noqa: F841


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.expanduser("~/work/nt_aot"))
    args = parser.parse_args()

    tensors = (
        Tensor(2, dtype=ninetoothed.bfloat16),
        Tensor(1, dtype=ninetoothed.bfloat16),
        Tensor(2, dtype=ninetoothed.bfloat16),
    )
    kernel = ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller="cuda",
        kernel_name="rms_norm_h3",
        output_dir=args.out,
        num_warps=2,
        num_stages=1,
    )
    print("[aot] kernel object:", type(kernel).__name__)

    for root, _, files in os.walk(args.out):
        for name in files:
            path = os.path.join(root, name)
            print(f"[file] {path}  {os.path.getsize(path)} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
