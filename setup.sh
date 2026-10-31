#!/usr/bin/env bash
# One-command setup for Google Colab or Kaggle (Linux + NVIDIA GPU).
# Usage: bash setup.sh
set -e

echo "== Installing Python dependencies =="
pip install -q -r requirements.txt

echo "== Fetching TripoSR code =="
if [ ! -d third_party/TripoSR ]; then
  git clone https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR
fi
# Pin to the commit you tested with, for reproducibility:
# (cd third_party/TripoSR && git checkout <commit>)

echo "== Installing TripoSR dependencies =="
# Skip TripoSR's exact pins (they clash with current Colab) and torchmcubes (built below).
grep -v -i -E "^(torch|torchmcubes|git\+)" third_party/TripoSR/requirements.txt \
  | sed 's/==.*//' > /tmp/triposr_reqs.txt
pip install -q -r /tmp/triposr_reqs.txt || echo "WARN: some TripoSR extras failed; continuing"

echo "== Building torchmcubes =="
pip install -q setuptools wheel cmake ninja
if pip install -q --no-build-isolation git+https://github.com/tatsy/torchmcubes.git; then
  echo "torchmcubes built OK"
else
  echo "WARN: torchmcubes build failed. Using scikit-image marching-cubes fallback (CPU, slightly slower)."
  pip install -q scikit-image
  cat > third_party/TripoSR/torchmcubes.py <<'PY'
# Fallback shim: same interface as torchmcubes.marching_cubes, backed by scikit-image.
import numpy as np
import torch
from skimage import measure


def marching_cubes(vol, thresh):
    v, f, _, _ = measure.marching_cubes(vol.detach().cpu().numpy(), level=float(thresh))
    return (
        torch.from_numpy(v.copy()).float().to(vol.device),
        torch.from_numpy(f.astype(np.int64)).to(vol.device),
    )
PY
fi

echo "== Done. Next: python scripts/gpu_check.py, then python scripts/selftest.py =="