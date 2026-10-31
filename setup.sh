#!/usr/bin/env bash
# One-command setup for Google Colab or Kaggle (Linux + NVIDIA GPU).
# Usage: bash setup.sh
set -e

echo "== Installing Python dependencies =="
pip install -q -r requirements.txt

echo "== Fetching TripoSR code (pinned in third_party/) =="
if [ ! -d third_party/TripoSR ]; then
  git clone https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR
fi
# TripoSR needs its own requirements. torchmcubes builds from source and needs a CUDA toolchain.
pip install -q -r third_party/TripoSR/requirements.txt || {
  echo "TripoSR requirements failed. See README Troubleshooting (torchmcubes build)."; exit 1; }

echo "== Done. Next: python scripts/gpu_check.py, then python scripts/selftest.py =="
