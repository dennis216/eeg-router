#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/mnt/d/EEGData/chbmit-1.0.0}"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"
wget -r -N -c -np --cut-dirs=3 -nH https://physionet.org/files/chbmit/1.0.0/
