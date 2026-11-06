"""探明九齿在"多输入元素级"场景下的访存效率规律。

对照 1/2/3/4 个输入 + 1 个输出的同形状 kernel，
看带宽随输入数的变化，判断是否存在系统性退化。
"""

import time

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

N = 5376


def make(n_inputs: int, bm: int = 1):
    B = block_size(bm, bm)

    def arr(*args):
        return tuple(a.tile((B, N)) for a in args)

    if n_inputs == 1:
        def app(a, y):
            y = a  # noqa: F841
    elif n_inputs == 2:
        def app(a, b, y):
            y = a + b  # noqa: F841
    elif n_inputs == 3:
        def app(a, b, c, y):
            y = a + b + c  # noqa: F841
    else:
        def app(a, b, c, d, y):
            y = a + b + c + d  # noqa: F841

    tensors = tuple(Tensor(2) for _ in range(n_inputs + 1))
    return ninetoothed.make(arr, app, tensors)


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
    peak = None

    # 峰值
    n = 512 * 1024 * 1024 // 2
    a0 = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    b0 = torch.empty_like(a0)
    tp = bench(lambda: b0.copy_(a0), 3, 10)
    peak = (2 * n * 2) / (tp / 1000) / 1e9
    print(f"peak BW {peak:.1f} GB/s\n")

    print(f"{'输入数':>6} {'耗时(ms)':>10} {'读+写趟':>8} {'带宽(GB/s)':>12} {'占峰值':>8}")
    for ni in (1, 2, 3, 4):
        xs = [torch.randn(seq, N, device="cuda", dtype=torch.bfloat16)
              for _ in range(ni)]
        y = torch.empty_like(xs[0])
        try:
            k = make(ni)
            k(*xs, y)
            torch.cuda.synchronize()
            t = bench(lambda: k(*xs, y))
            moves = (ni + 1) * seq * N * 2
            gb = moves / (t / 1000) / 1e9
            print(f"{ni:>6} {t:>10.4f} {ni+1:>8} {gb:>12.1f} {gb/peak*100:>7.0f}%")
        except Exception as e:
            print(f"{ni:>6} FAIL {str(e)[:70]}")

    # torch 对照
    print()
    for ni in (1, 2, 3, 4):
        xs = [torch.randn(seq, N, device="cuda", dtype=torch.bfloat16)
              for _ in range(ni)]
        o = torch.empty_like(xs[0])
        if ni == 1:
            fn = lambda: o.copy_(xs[0])
        elif ni == 2:
            fn = lambda: torch.add(xs[0], xs[1], out=o)
        elif ni == 3:
            fn = lambda: torch.add(torch.add(xs[0], xs[1]), xs[2], out=o)
        else:
            fn = lambda: torch.add(torch.add(xs[0], xs[1]),
                                    torch.add(xs[2], xs[3]), out=o)
        t = bench(fn)
        moves = (ni + 1) * seq * N * 2
        gb = moves / (t / 1000) / 1e9
        print(f"torch n={ni}  {t:.4f} ms  {gb:.1f} GB/s  ({gb/peak*100:.0f}% 峰值)")


if __name__ == "__main__":
    main()
