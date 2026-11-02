#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00-env.sh"
python -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.head_exchange.benchmark --transport fused --rows 1 7 9 129 --ratios 0 1 2 --context 32768 --vary-lengths --repeat 3 --inner 2 --interface build/flash_mla_interface.py --output "$RESULT_DIR/tp8-vary-lengths.json" 2>&1 | tee "$RESULT_DIR/tp8-vary-lengths.log"
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -O3 -std=c++17 -arch=sm_90a --shared -Xcompiler -fPIC -cudart shared -Xlinker -rpath -Xlinker "${CUDA_HOME:-/usr/local/cuda}/lib64" experiments/segment_exchange_antgrid.cu -o build/libsegment_exchange_antgrid.so
export DSV41_HEAD_EXCHANGE_LIBRARY="$PROJECT_ROOT/build/libsegment_exchange_antgrid.so"
python -m torch.distributed.run --standalone --nproc-per-node=8 scripts/05-grid-scan.py 2>&1 | tee "$RESULT_DIR/tp8-grid-scan.log"
