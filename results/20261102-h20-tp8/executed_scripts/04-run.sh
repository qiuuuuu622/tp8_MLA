#!/usr/bin/env bash
set -euo pipefail
source /root/h20-tp8-20261102/scripts/00-env.sh
trap 'rc=$?; echo "exit_code=$rc utc=$(date -u +%FT%TZ)" > results/runner.status' EXIT
timeout -k 15 600 bash scripts/01-verify.sh > results/tp8-stress-final.log 2>&1
timeout -k 15 1800 bash scripts/03-function-ab.sh
