#!/usr/bin/env bash
# cluster side: run single-op benchmark
# usage: srun ... bash scripts/run_bench.sh [extra args]
set -euo pipefail

source "$HOME/setup_env.sh"
cd "$HOME/work/infinilm"

SEQ="${SEQ:-2048,8192}"
OPS="${OPS:-rms_norm,silu_and_mul,linear}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"   # HYGON maps to "cuda" in infinicore python API

python bench/bench_op.py \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seqs "$SEQ" \
  --ops "$OPS" \
  --out "results/bench_${DTYPE}_$(date +%H%M%S).json" \
  "$@"
