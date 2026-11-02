#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00-env.sh"
for transport in nccl fused; do
  python -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.head_exchange.benchmark --transport "$transport" --rows 1 8 32 64 96 128 192 288 384 512 --ratios 0 1 2 --context 32768 --repeat 30 --inner 20 --interface build/flash_mla_interface.py --output "$RESULT_DIR/tp8-$transport.json" 2>&1 | tee "$RESULT_DIR/tp8-$transport.log"
done
python -m benchmarks.kernels.head_exchange.summarize --fused "$RESULT_DIR/tp8-fused.json" --nccl "$RESULT_DIR/tp8-nccl.json" --output "$RESULT_DIR/tp8-comparison.csv"
