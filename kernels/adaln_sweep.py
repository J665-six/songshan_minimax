"""AdaLN 融合 kernel 的 BLOCK_M 扫描，验证是否只是参数没调好。"""

import time

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

N = 5376


def make(bm):
    B = block_size(bm, bm)

    def arr(x, s, h, y):
        return (x.tile((B, N)), s.tile((B, N)),
                h.tile((B, N)), y.tile((B, N)))

    def app(x, s, h, y):
        y = x + x * s + h  # noqa: F841

    return ninetoothed.make(
        arr, app, (Tensor(2), Tensor(2), Tensor(2), Tensor(2))
    )


def bench(fn, w=10, i=50):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(i):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / i * 1000


def main():
    seq = 8192
    x = torch.randn(seq, N, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(seq, N, device="cuda", dtype=torch.bfloat16)
    h = torch.randn(seq, N, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(x)

    for bm in (1, 2, 4):
        try:
            k = make(bm)
            k(x, s, h, o)
            torch.cuda.synchronize()
            t = bench(lambda: k(x, s, h, o))
            print(f"BM={bm}   {t:.4f} ms   {4*seq*N*2/(t/1000)/1e9:.1f} GB/s")
        except Exception as e:
            print(f"BM={bm}   FAIL {str(e)[:90]}")

    t = bench(lambda: torch.add(torch.add(x, torch.mul(x, s)), h))
    print(f"torch 3-op  {t:.4f} ms   {9*seq*N*2/(t/1000)/1e9:.1f} GB/s(等效)")


if __name__ == "__main__":
    main()
