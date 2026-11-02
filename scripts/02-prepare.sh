#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00-env.sh"
curl -fL https://raw.githubusercontent.com/deepseek-ai/FlashMLA/0eee43b12f034b657133cf2afca6a72ebb6efccf/flash_mla/flash_mla_interface.py -o build/flash_mla_interface.py
python - <<'PY'
from pathlib import Path
import hashlib
assert hashlib.sha256(Path('build/flash_mla_interface.py').read_bytes()).hexdigest() == '76eb66a7922531f402ce6fadd51e01aab19b84bff063d2b33dfbaa68f8915e26'
PY
python -m vllm_dsv41_head_exchange.build --nvcc "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
