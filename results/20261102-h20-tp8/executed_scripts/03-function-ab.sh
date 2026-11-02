#!/usr/bin/env bash
set -euo pipefail
source /root/h20-tp8-20261102/scripts/00-env.sh
for transport in nccl fused; do
  python -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.head_exchange.benchmark --transport "$transport" --rows 1 8 32 64 96 128 192 288 384 512 --ratios 0 1 2 --context 32768 --repeat 30 --inner 20 --interface flash_mla_interface.py --output "results/tp8-$transport.json" > "results/tp8-$transport.log" 2>&1
done
python -m benchmarks.kernels.head_exchange.summarize --fused results/tp8-fused.json --nccl results/tp8-nccl.json --output results/tp8-comparison.csv
