#!/usr/bin/env bash
set -euo pipefail
source /root/h20-tp8-20261102/scripts/00-env.sh
export PYTHONPATH=$PWD
trap 'rc=$?; echo "exit_code=$rc utc=$(date -u +%FT%TZ)" > results/additional.status' EXIT
python -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.head_exchange.benchmark --transport fused --rows 1 7 9 129 --ratios 0 1 2 --context 32768 --vary-lengths --repeat 3 --inner 2 --interface flash_mla_interface.py --output results/tp8-vary-lengths.json > results/tp8-vary-lengths.log 2>&1
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_90a --shared -Xcompiler -fPIC -cudart shared -Xlinker -rpath -Xlinker /usr/local/cuda/lib64 experiments/segment_exchange_antgrid.cu -o experiments/libsegment_exchange_antgrid.so > results/antgrid-build.log 2>&1
export DSV41_HEAD_EXCHANGE_LIBRARY=$PWD/experiments/libsegment_exchange_antgrid.so
python -m torch.distributed.run --standalone --nproc-per-node=8 scripts/05-grid-scan.py > results/tp8-grid-scan.log 2>&1
