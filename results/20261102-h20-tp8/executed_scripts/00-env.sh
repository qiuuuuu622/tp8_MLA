#!/usr/bin/env bash
set -euo pipefail
cd /root/h20-tp8-20261102
export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH="$PWD/.venv/bin:$PATH"
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
