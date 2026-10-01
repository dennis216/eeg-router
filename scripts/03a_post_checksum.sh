#!/usr/bin/env bash
set -euo pipefail

checksum_pid="${1:?checksum PID required}"
data_root="${2:-/mnt/d/EEGData/chbmit-1.0.0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while kill -0 "$checksum_pid" 2>/dev/null; do
  sleep 10
done

total="$(wc -l < "$data_root/SHA256SUMS.txt")"
ok="$(grep -c ': OK$' "$repo_root/logs/chbmit_sha256.log")"
if [[ "$ok" -ne "$total" ]]; then
  echo "CHECKSUM_GATE_FAILED ok=$ok total=$total"
  exit 1
fi

cd "$repo_root"
source .venv/bin/activate
python scripts/03_build_manifests.py --root "$data_root"
python -m bfa.data.split --seeds 17 42 3407 2026 777 --out manifests/splits
echo "POST_CHECKSUM_PIPELINE_OK"
