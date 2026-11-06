"""单算子基准：InfiniCore(Hygon DCU) vs PyTorch(HIP)

在集群作业里运行：
    python bench/bench_op.py --out results/bench.json

两个关键点：
1. InfiniCore 的 python 设备名里 HYGON 映射为 "cuda"（见 device.py 的 _TORCH_DEVICE_MAP）。
2. `from_torch` 是零拷贝（内部走 from_blob + data_ptr），因此用 out= 传入
   torch 张量包装后的 infinicore tensor，结果可直接在 torch 侧读取对比。
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import infinicore
from infinicore.tensor import from_torch
import infinicore.nn.functional as icf
from infinicore.ops.mha import mha as ic_mha
from infinicore.ops.add_rms_norm import add_rms_norm as ic_add_rms_norm

try:
    from bench import cases
except ImportError:  # 直接以脚本方式运行
    import cases  # type: ignore


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def sync():
    try:
        infinicore.sync_device()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters * 1000.0


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = (af - bf).abs()
    denom = bf.abs().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item() if af.numel() > 1 else 1.0
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / denom).max().item(),
        "mean_abs": diff.mean().item(),
        "cosine": cos,
    }


def run_case(name: str, dtype: torch.dtype, seq: int, builder) -> dict:
    torch_fn, ic_fn, torch_out, ic_out = builder(seq, dtype, torch.device("cuda"))

    torch_fn()
    ic_fn()
    sync()
    stats = compare(ic_out.detach().clone(), torch_out.detach().clone())

    t_torch = bench(torch_fn)
    t_ic = bench(ic_fn)

    return {
        "op": name,
        "seq": seq,
        "dtype": str(dtype).split(".")[-1],
        "torch_ms": round(t_torch, 4),
        "infinicore_ms": round(t_ic, 4),
        "speedup_ic_over_torch": round(t_torch / t_ic, 3) if t_ic > 0 else None,
        "tokens_per_s": round(seq / (t_ic / 1000.0), 1) if t_ic > 0 else None,
        "accuracy": {k: float(f"{v:.3e}") for k, v in stats.items()},
    }


# ---------------- 各算子构造器 ----------------

def case_rms_norm(seq, dtype, dev):
    x = torch.randn(seq, cases.HIDDEN, device=dev, dtype=dtype)
    w = torch.randn(cases.HIDDEN, device=dev, dtype=dtype)
    out_t = torch.empty_like(x)
    out_i = torch.empty_like(x)
    icx, icw, ico = from_torch(x), from_torch(w), from_torch(out_i)

    def torch_fn():
        out_t.copy_(torch.nn.functional.rms_norm(x, [cases.HIDDEN], w, 1e-5))

    def ic_fn():
        icf.rms_norm(icx, [cases.HIDDEN], icw, 1e-5, out=ico)

    return torch_fn, ic_fn, out_t, out_i


def case_add_rms_norm(seq, dtype, dev):
    a = torch.randn(seq, cases.HIDDEN, device=dev, dtype=dtype)
    b = torch.randn(seq, cases.HIDDEN, device=dev, dtype=dtype)
    w = torch.randn(cases.HIDDEN, device=dev, dtype=dtype)
    out_t = torch.empty_like(a)
    res_t = torch.empty_like(a)
    out_i = torch.empty_like(a)
    res_i = torch.empty_like(a)
    ica, icb, icw = from_torch(a), from_torch(b), from_torch(w)
    ico, icr = from_torch(out_i), from_torch(res_i)

    def torch_fn():
        s = a + b
        res_t.copy_(s)
        out_t.copy_(torch.nn.functional.rms_norm(s, [cases.HIDDEN], w, 1e-5))

    def ic_fn():
        ic_add_rms_norm(ica, icb, icw, 1e-5, out=ico, residual=icr)

    return torch_fn, ic_fn, out_t, out_i


def case_silu_and_mul(seq, dtype, dev):
    x = torch.randn(seq, cases.FFN_INNER, device=dev, dtype=dtype)
    out_t = torch.empty(seq, cases.FFN, device=dev, dtype=dtype)
    out_i = torch.empty(seq, cases.FFN, device=dev, dtype=dtype)
    icx, ico = from_torch(x), from_torch(out_i)

    def torch_fn():
        gate, up = x.chunk(2, dim=-1)
        out_t.copy_(torch.nn.functional.silu(gate) * up)

    def ic_fn():
        icf.silu_and_mul(icx, out=ico)

    return torch_fn, ic_fn, out_t, out_i


def case_linear(seq, dtype, dev):
    x = torch.randn(seq, cases.HIDDEN, device=dev, dtype=dtype)
    w = torch.randn(cases.FFN, cases.HIDDEN, device=dev, dtype=dtype)
    out_t = torch.empty(seq, cases.FFN, device=dev, dtype=dtype)
    out_i = torch.empty(seq, cases.FFN, device=dev, dtype=dtype)
    icx, icw, ico = from_torch(x), from_torch(w), from_torch(out_i)

    def torch_fn():
        out_t.copy_(torch.nn.functional.linear(x, w))

    def ic_fn():
        icf.linear(icx, icw, out=ico)

    return torch_fn, ic_fn, out_t, out_i


def case_attention(seq, dtype, dev):
    q = torch.randn(1, seq, cases.HEADS, cases.HEAD_DIM, device=dev, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out_t = torch.empty_like(q)
    out_i = torch.empty_like(q)
    icq, ick, icv, ico = from_torch(q), from_torch(k), from_torch(v), from_torch(out_i)
    scale = 1.0 / (cases.HEAD_DIM ** 0.5)

    def torch_fn():
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        out_t.copy_(out.transpose(1, 2))

    def ic_fn():
        ic_mha(icq, ick, icv, None, scale, False, out=ico)

    return torch_fn, ic_fn, out_t, out_i


CASES = {
    "rms_norm": case_rms_norm,
    "add_rms_norm": case_add_rms_norm,
    "silu_and_mul": case_silu_and_mul,
    "linear": case_linear,
    "attention": case_attention,
}


def measure_peak_bandwidth(mb: int = 512) -> float:
    """用 device-to-device 拷贝估算有效显存带宽 (GB/s)。"""
    elems = mb * 1024 * 1024 // 2
    a = torch.empty(elems, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a), warmup=3, iters=10)
    bytes_moved = 2 * elems * 2  # 读 + 写
    return bytes_moved / (ms / 1000.0) / 1e9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda",
                        help="InfiniCore python device type (HYGON maps to cuda)")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    parser.add_argument("--seqs", default=",".join(str(s) for s in cases.SEQ_LENS))
    parser.add_argument("--ops", default=",".join(CASES))
    parser.add_argument("--out", default="results/bench.json")
    args = parser.parse_args()

    infinicore.set_device(infinicore.device(args.device, 0))
    dtype = DTYPES[args.dtype]
    results, errors = [], []

    print(f"[env] torch={torch.__version__} hip={torch.version.hip} "
          f"devices={torch.cuda.device_count()} dtype={args.dtype}")

    peak_bw = measure_peak_bandwidth()
    print(f"[env] peak copy bandwidth ≈ {peak_bw:.1f} GB/s")

    for seq in [int(s) for s in args.seqs.split(",")]:
        for name in args.ops.split(","):
            builder = CASES.get(name)
            if builder is None:
                continue
            try:
                row = run_case(name, dtype, seq, builder)
                results.append(row)
                print(f"[ok] {name:<14} seq={seq:<6} ic={row['infinicore_ms']:>8.3f}ms "
                      f"torch={row['torch_ms']:>8.3f}ms  x{row['speedup_ic_over_torch']}  "
                      f"cos={row['accuracy']['cosine']:.6f}")
            except Exception as exc:  # noqa: BLE001
                errors.append({"op": name, "seq": seq, "error": str(exc),
                               "trace": traceback.format_exc()[-800:]})
                print(f"[fail] {name} seq={seq}: {exc}")

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": args.device,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "peak_bandwidth_gbs": round(peak_bw, 1),
        "shapes": cases.h3_summary(),
        "results": results,
        "errors": errors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[write] {out}  ({len(results)} ok, {len(errors)} failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
