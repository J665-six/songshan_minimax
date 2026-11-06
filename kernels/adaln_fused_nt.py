"""NineToothed 融合 AdaLN 调制算子。

参考实现（InfiniLM C++，minimax_h3_transformer.cpp:54）：

    auto scaled = op::mul(normalized, scale);
    return op::add(op::add(normalized, scaled), shift);

数学上等价于：
    y = normalized * (1 + scale) + shift

当前实现的问题：
    1. 三次 kernel 启动（mul, add, add）
    2. 两个中间临时张量（scaled, normalized+scaled），峰值多占 2×tensor
    3. 四次全量访存（读 n、读 n、读 scale、写 tmp1、读 tmp1、读 n、写 tmp2…）

融合后：
    - 一次读 normalized、scale、shift，一次写 y
    - 访存从 6 趟降到 2 趟（读 3 + 写 1，按元素计）

这个优化不改变任何数值语义——公式完全一致，只是不再分步落盘。
"""

from __future__ import annotations

import argparse
import time

import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

HIDDEN = 5376
BM = block_size(1, 1)


def arrangement(x, scale, shift, y):
    return (
        x.tile((BM, HIDDEN)),
        scale.tile((BM, HIDDEN)),
        shift.tile((BM, HIDDEN)),
        y.tile((BM, HIDDEN)),
    )


def application(x, scale, shift, y):
    # y = x * (1 + scale) + shift
    y = x + x * scale + shift  # noqa: F841


def build():
    return ninetoothed.make(
        arrangement, application,
        (Tensor(2), Tensor(2), Tensor(2), Tensor(2)),
    )


def ref_fused(x, scale, shift):
    """融合公式的 fp32 参考。"""
    xf, sf, hf = x.float(), scale.float(), shift.float()
    return (xf + xf * sf + hf).to(x.dtype)


def ref_baseline(x, scale, shift):
    """逐算子参考：完全复刻 C++ 的三步实现，用于确认两者等价。"""
    scaled = (x.float() * scale.float()).to(x.dtype)
    t1 = (x.float() + scaled.float()).to(x.dtype)
    t2 = (t1.float() + shift.float()).to(x.dtype)
    return t2


def bench(fn, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def gbs(nbytes, ms):
    return nbytes / (ms / 1000.0) / 1e9


def run(seq: int, dtype, peak: float):
    print("=" * 72)
    print(f"  AdaLN 融合消融  |  seq={seq}  dtype={dtype}")
    print("=" * 72)

    g = torch.Generator(device="cuda").manual_seed(20261030)
    x = torch.randn(seq, HIDDEN, device="cuda", dtype=dtype, generator=g)
    scale = torch.randn(seq, HIDDEN, device="cuda", dtype=dtype, generator=g) * 0.1
    shift = torch.randn(seq, HIDDEN, device="cuda", dtype=dtype, generator=g) * 0.1

    # ---------- 基线：三次 torch 调用，复刻 C++ 行为 ----------
    def baseline():
        scaled = torch.mul(x, scale)
        t1 = torch.add(x, scaled)
        return torch.add(t1, shift)

    t_base = [bench(baseline) for _ in range(3)]
    out_base = baseline()
    print("\n[基线] mul → add → add（三次 kernel）")
    # 基线访存：mul 读2写1，add 读2写1，add 读2写1 = 9 趟
    nb_base = 9 * seq * HIDDEN * 2
    for i, t in enumerate(t_base, 1):
        print(f"   第 {i} 轮  {t:8.4f} ms   {gbs(nb_base, t):7.1f} GB/s(等效)")
    print(f"   中位     {sorted(t_base)[1]:8.4f} ms")

    # ---------- 九齿融合 ----------
    kernel = build()
    out_nt = torch.empty_like(x)

    def fused():
        kernel(x, scale, shift, out_nt)

    fused()
    torch.cuda.synchronize()
    t_nt = [bench(fused) for _ in range(3)]
    print("\n[优化] NineToothed 融合 affine（一次 kernel）")
    # 融合访存：读 x/scale/shift + 写 y = 4 趟
    nb_nt = 4 * seq * HIDDEN * 2
    for i, t in enumerate(t_nt, 1):
        print(f"   第 {i} 轮  {t:8.4f} ms   {gbs(nb_nt, t):7.1f} GB/s")
    print(f"   中位     {sorted(t_nt)[1]:8.4f} ms")

    # ---------- 正确性 ----------
    ref = ref_fused(x, scale, shift)
    def stats(a, b):
        af, bf = a.float().flatten(), b.float().flatten()
        d = (af - bf).abs()
        return (torch.nn.functional.cosine_similarity(af, bf, dim=0).item(),
                d.max().item(), (d / bf.abs().clamp_min(1e-6)).max().item())

    cb, ab, rb = stats(out_base, ref)
    cn, an, rn = stats(out_nt, ref)
    print("\n[正确性] 与 fp32 参考对比")
    print(f"   {'实现':10} {'cos':>13} {'max_abs':>12} {'max_rel':>12}")
    print(f"   {'基线':10} {cb:13.8f} {ab:12.3e} {rb:12.3e}")
    print(f"   {'九齿融合':10} {cn:13.8f} {an:12.3e} {rn:12.3e}")

    # 两个实现互相等价性
    ce, ae, _ = stats(out_nt, out_base)
    print(f"\n   [等价性] 九齿 vs 基线: cos={ce:.8f}  max_abs={ae:.3e}")

    mb, mn = sorted(t_base)[1], sorted(t_nt)[1]
    print("\n[结论]")
    print(f"   加速比     {mb/mn:.3f}×")
    print(f"   基线带宽   {gbs(nb_base, mb):7.1f} GB/s  ({gbs(nb_base, mb)/peak*100:.0f}% 峰值)")
    print(f"   融合带宽   {gbs(nb_nt, mn):7.1f} GB/s  ({gbs(nb_nt, mn)/peak*100:.0f}% 峰值)")
    print(f"   单次省时   {mb-mn:.4f} ms")
    print(f"\n   推算到真实推理：")
    print(f"     每层 2 次 AdaLN × 50 层 × 50 步 = 5000 次")
    print(f"     若序列长 111600（720p/124帧），单次省时 × 比例估算见文档")

    return {"seq": seq, "dtype": str(dtype).split(".")[-1],
            "baseline_ms": mb, "fused_ms": mn, "speedup": mb / mn,
            "baseline_runs": t_base, "fused_runs": t_nt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="8192")
    ap.add_argument("--dtypes", default="bf16")
    args = ap.parse_args()

    print(f"torch {torch.__version__} hip {torch.version.hip} "
          f"dcu {torch.cuda.device_count()} {torch.cuda.get_device_name(0)}")

    # 峰值带宽
    n = 512 * 1024 * 1024 // 2
    a = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    peak = (2 * n * 2) / (bench(lambda: b.copy_(a), 3, 10) / 1000.0) / 1e9
    print(f"peak BW {peak:.1f} GB/s\n")

    dmap = {"bf16": torch.bfloat16, "fp16": torch.float16}
    for dt in args.dtypes.split(","):
        for s in [int(v) for v in args.seqs.split(",")]:
            run(s, dmap[dt], peak)
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
