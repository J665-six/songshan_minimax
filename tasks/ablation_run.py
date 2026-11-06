"""消融实验 harness：同一个脚本，只切换算子实现。

设计原则：
- 同一份输入张量（seed 固定）
- 同一台 DCU、同一个作业
- 同一套计时函数
- 同一份数值参考
- 唯一变量：被优化的算子实现（基线 vs 九齿）

这样得到的差值只能归因于算子替换。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

import infinicore
from infinicore.tensor import from_torch
import infinicore.nn.functional as icf
from infinicore.ops.add_rms_norm import add_rms_norm as ic_add_rms_norm

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

# ---------------- 实验配置 ----------------

HIDDEN = 5376          # H3 DiT hidden_size
EPS = 1e-5
SEED = 20261030


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72, flush=True)


def env_report() -> dict:
    info = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "devices": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
        "python": platform.python_version(),
    }
    try:
        out = subprocess.run(["hy-smi", "--showproductname"],
                             capture_output=True, text=True, timeout=10)
        info["hy_smi"] = out.stdout.strip().splitlines()[-3:][0] if out.stdout else "n/a"
    except Exception:
        info["hy_smi"] = "n/a"
    for k, v in info.items():
        print(f"  {k:14} {v}")
    return info


def sync():
    try:
        infinicore.sync_device()
    except Exception:
        pass
    torch.cuda.synchronize()


def bench(fn, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1000.0


def stable_bench(fn, rounds=3, **kw) -> list[float]:
    return [bench(fn, **kw) for _ in range(rounds)]


def gbs(bytes_moved: float, ms: float) -> float:
    """字节数 / 毫秒 → GB/s。"""
    return bytes_moved / (ms / 1000.0) / 1e9


def cmp_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    af, bf = a.float().flatten(), b.float().flatten()
    diff = (af - bf).abs()
    return {
        "cosine": torch.nn.functional.cosine_similarity(af, bf, dim=0).item(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / bf.abs().clamp_min(1e-6)).max().item(),
    }


def peak_bandwidth(mb=512) -> float:
    n = mb * 1024 * 1024 // 2
    a = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    t = bench(lambda: b.copy_(a), warmup=3, iters=10)
    return (2 * n * 2) / (t / 1000.0) / 1e9


# ---------------- 九齿 kernel ----------------

N_CONST = HIDDEN
BM = block_size(1, 1)


def _rms_arrangement(x, w, y):
    return (x.tile((BM, N_CONST)),
            w.tile((N_CONST,)).expand((x.shape[0], -1)),
            y.tile((BM, N_CONST)))


def _rms_application(x, w, y):
    ms = ntl.sum(x * x, axis=1) / x.shape[1]
    inv = 1.0 / ntl.sqrt(ms + 1e-5)
    y = x * inv[:, None] * w  # noqa: F841


def _add_rms_arrangement(a, b, w, y, res):
    return (a.tile((BM, N_CONST)),
            b.tile((BM, N_CONST)),
            w.tile((N_CONST,)).expand((a.shape[0], -1)),
            y.tile((BM, N_CONST)),
            res.tile((BM, N_CONST)))


def _add_rms_application(a, b, w, y, res):
    s = a + b
    ms = ntl.sum(s * s, axis=1) / s.shape[1]
    inv = 1.0 / ntl.sqrt(ms + 1e-5)
    y = s * inv[:, None] * w  # noqa: F841
    res = s  # noqa: F841


def build_rms():
    return ninetoothed.make(_rms_arrangement, _rms_application,
                            (Tensor(2), Tensor(1), Tensor(2)))


def build_add_rms():
    return ninetoothed.make(_add_rms_arrangement, _add_rms_application,
                            (Tensor(2), Tensor(2), Tensor(1), Tensor(2), Tensor(2)))


# ---------------- 参考实现 ----------------

def torch_rms_ref(x, w):
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf / torch.sqrt(ms + EPS) * w.float()).to(x.dtype)


def torch_add_rms_ref(a, b, w):
    s = a.float() + b.float()
    ms = s.pow(2).mean(dim=-1, keepdim=True)
    return (s / torch.sqrt(ms + EPS) * w.float()).to(a.dtype), s.to(a.dtype)


# ---------------- 实验 1：RMSNorm ----------------

def experiment_rms(seq: int, dtype, peak: float) -> dict:
    banner(f"实验 1：RMSNorm 消融  |  seq={seq}  dtype={dtype}")

    g = torch.Generator(device="cuda").manual_seed(SEED)
    x = torch.randn(seq, HIDDEN, device="cuda", dtype=dtype, generator=g)
    w = torch.randn(HIDDEN, device="cuda", dtype=dtype, generator=g)

    # --- 基线 ---
    out_base = torch.empty_like(x)
    icx, icw, ico = from_torch(x), from_torch(w), from_torch(out_base)

    def run_base():
        icf.rms_norm(icx, [HIDDEN], icw, EPS, out=ico)

    run_base(); sync()
    print("\n[基线] InfiniCore rms_norm")
    t_base = stable_bench(run_base)
    nbytes = 2 * seq * HIDDEN * 2
    for i, t in enumerate(t_base, 1):
        print(f"   第 {i} 轮  {t:8.4f} ms   {gbs(nbytes, t):7.1f} GB/s")
    print(f"   中位     {sorted(t_base)[1]:8.4f} ms")

    # --- 九齿（直接吃 torch 张量；与基线共享同一块显存）---
    kernel = build_rms()
    out_nt = torch.empty_like(x)

    def run_nt():
        kernel(x, w, out_nt)

    run_nt(); sync()
    print("\n[优化] NineToothed rms_norm")
    t_nt = stable_bench(run_nt)
    for i, t in enumerate(t_nt, 1):
        print(f"   第 {i} 轮  {t:8.4f} ms   {gbs(nbytes, t):7.1f} GB/s")
    print(f"   中位     {sorted(t_nt)[1]:8.4f} ms")

    # --- 正确性 ---
    ref = torch_rms_ref(x, w)
    s_base, s_nt = cmp_stats(out_base, ref), cmp_stats(out_nt, ref)
    print("\n[正确性] 与 PyTorch fp32 参考对比")
    print(f"   {'实现':8} {'cos':>12} {'max_abs':>12} {'max_rel':>12}")
    for name, s in (("基线", s_base), ("九齿", s_nt)):
        print(f"   {name:8} {s['cosine']:12.8f} {s['max_abs']:12.3e} {s['max_rel']:12.3e}")

    mb = sorted(t_base)[1]
    mn = sorted(t_nt)[1]
    print("\n[结论]")
    print(f"   加速比     {mb/mn:.3f}×")
    print(f"   基线带宽   {gbs(nbytes, mb):7.1f} GB/s  ({gbs(nbytes, mb)/peak*100:.0f}% 峰值)")
    print(f"   九齿带宽   {gbs(nbytes, mn):7.1f} GB/s  ({gbs(nbytes, mn)/peak*100:.0f}% 峰值)")

    return {
        "experiment": "rms_norm", "seq": seq,
        "dtype": str(dtype).split(".")[-1],
        "baseline_ms": mb, "ninetoothed_ms": mn,
        "speedup": mb / mn,
        "baseline_gbs": gbs(nbytes, mb),
        "ninetoothed_gbs": gbs(nbytes, mn),
        "baseline_runs": t_base, "ninetoothed_runs": t_nt,
        "accuracy_baseline": s_base, "accuracy_ninetoothed": s_nt,
    }


# ---------------- 实验 2：Add-RMSNorm ----------------

def experiment_add_rms(seq: int, dtype, peak: float) -> dict:
    banner(f"实验 2：Add-RMSNorm 消融  |  seq={seq}  dtype={dtype}")

    g = torch.Generator(device="cuda").manual_seed(SEED + 1)
    a = torch.randn(seq, HIDDEN, device="cuda", dtype=dtype, generator=g)
    b = torch.randn(seq, HIDDEN, device="cuda", dtype=dtype, generator=g)
    w = torch.randn(HIDDEN, device="cuda", dtype=dtype, generator=g)

    y_base = torch.empty_like(a); r_base = torch.empty_like(a)
    ica, icb, icw = from_torch(a), from_torch(b), from_torch(w)
    icy, icr = from_torch(y_base), from_torch(r_base)

    def run_base():
        ic_add_rms_norm(ica, icb, icw, EPS, out=icy, residual=icr)

    run_base(); sync()
    print("\n[基线] InfiniCore add_rms_norm")
    t_base = stable_bench(run_base)
    nbytes = 4 * seq * HIDDEN * 2
    for i, t in enumerate(t_base, 1):
        print(f"   第 {i} 轮  {t:8.4f} ms   {gbs(nbytes, t):7.1f} GB/s")
    print(f"   中位     {sorted(t_base)[1]:8.4f} ms")

    kernel = build_add_rms()
    y_nt = torch.empty_like(a); r_nt = torch.empty_like(a)

    def run_nt():
        kernel(a, b, w, y_nt, r_nt)

    run_nt(); sync()
    print("\n[优化] NineToothed add_rms_norm")
    t_nt = stable_bench(run_nt)
    for i, t in enumerate(t_nt, 1):
        print(f"   第 {i} 轮  {t:8.4f} ms   {gbs(nbytes, t):7.1f} GB/s")
    print(f"   中位     {sorted(t_nt)[1]:8.4f} ms")

    ref_y, ref_r = torch_add_rms_ref(a, b, w)
    s_base, s_nt = cmp_stats(y_base, ref_y), cmp_stats(y_nt, ref_y)
    s_base_r, s_nt_r = cmp_stats(r_base, ref_r), cmp_stats(r_nt, ref_r)
    print("\n[正确性] 与 PyTorch fp32 参考对比")
    print(f"   {'实现':8} {'cos(y)':>12} {'cos(res)':>12} {'max_abs':>12} {'max_rel':>12}")
    print(f"   {'基线':8} {s_base['cosine']:12.8f} {s_base_r['cosine']:12.8f} "
          f"{s_base['max_abs']:12.3e} {s_base['max_rel']:12.3e}")
    print(f"   {'九齿':8} {s_nt['cosine']:12.8f} {s_nt_r['cosine']:12.8f} "
          f"{s_nt['max_abs']:12.3e} {s_nt['max_rel']:12.3e}")

    mb, mn = sorted(t_base)[1], sorted(t_nt)[1]
    print("\n[结论]")
    print(f"   加速比     {mb/mn:.3f}×")
    print(f"   基线带宽   {gbs(nbytes, mb):7.1f} GB/s  ({gbs(nbytes, mb)/peak*100:.0f}% 峰值)")
    print(f"   九齿带宽   {gbs(nbytes, mn):7.1f} GB/s  ({gbs(nbytes, mn)/peak*100:.0f}% 峰值)")

    return {
        "experiment": "add_rms_norm", "seq": seq,
        "dtype": str(dtype).split(".")[-1],
        "baseline_ms": mb, "ninetoothed_ms": mn, "speedup": mb / mn,
        "baseline_gbs": gbs(nbytes, mb),
        "ninetoothed_gbs": gbs(nbytes, mn),
        "baseline_runs": t_base, "ninetoothed_runs": t_nt,
        "accuracy_baseline": s_base, "accuracy_ninetoothed": s_nt,
        "accuracy_res_baseline": s_base_r, "accuracy_res_ninetoothed": s_nt_r,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="8192")
    ap.add_argument("--dtypes", default="bf16,fp16")
    ap.add_argument("--out", default="results/ablation.json")
    args = ap.parse_args()

    banner("环境")
    env = env_report()
    infinicore.set_device(infinicore.device("cuda", 0))
    peak = peak_bandwidth()
    print(f"  {'peak BW':14} {peak:.1f} GB/s  (device-to-device copy)")

    dmap = {"bf16": torch.bfloat16, "fp16": torch.float16}
    results = []
    for dt in args.dtypes.split(","):
        for s in [int(v) for v in args.seqs.split(",")]:
            try:
                results.append(experiment_rms(s, dmap[dt], peak))
            except Exception as e:
                print(f"[FAIL] rms seq={s} {dt}: {e}")
            try:
                results.append(experiment_add_rms(s, dmap[dt], peak))
            except Exception as e:
                print(f"[FAIL] add_rms seq={s} {dt}: {e}")

    banner("汇总")
    print(f"   {'算子':16} {'seq':>7} {'dtype':>6} {'基线':>10} {'九齿':>10} {'加速':>8} {'cos':>10}")
    for r in results:
        print(f"   {r['experiment']:16} {r['seq']:>7} {r['dtype']:>6} "
              f"{r['baseline_ms']:10.4f} {r['ninetoothed_ms']:10.4f} "
              f"{r['speedup']:7.2f}× {r['accuracy_ninetoothed']['cosine']:10.8f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": env, "peak_gbs": peak, "results": results},
                              indent=2, ensure_ascii=False))
    print(f"\n[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
