#!/usr/bin/env bash
set -euo pipefail
source /root/h20-tp8-20261102/scripts/00-env.sh
python -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.head_exchange.verify --rows 1 7 8 9 32 64 127 128 129 256 512 --iterations 200 --output results/tp8-stress.json
