#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export RESULT_DIR="$PROJECT_ROOT/build/tp8-results"
mkdir -p "$RESULT_DIR"
