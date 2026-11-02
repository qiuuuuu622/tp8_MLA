#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00-env.sh"
python -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.head_exchange.verify --rows 1 7 8 9 32 64 127 128 129 256 512 --iterations 200 --output "$RESULT_DIR/tp8-stress.json" 2>&1 | tee "$RESULT_DIR/tp8-stress.log"
