#!/usr/bin/env bash
# Setup for running GraphCast (DeepMind) on remote Ubuntu 22.04 with RTX 4090.
# Run once on the remote machine (e.g. after SSH), then use run_graphcast_remote.py

set -e

echo "=== GraphCast remote setup (Ubuntu 22.04, RTX 4090) ==="

# 1. Check NVIDIA driver (need 525+ for CUDA 12 / RTX 4090)
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. Install NVIDIA driver (e.g. sudo apt install nvidia-driver-535)."
fi

# 2. Create virtual environment (use venv or conda)
PYDIR="${PYDIR:-./venv_graphcast}"
if [[ -n "$CONDA_PREFIX" ]]; then
  echo "Using conda: $CONDA_PREFIX"
  pip install -U pip
else
  echo "Creating venv at $PYDIR"
  python3 -m venv "$PYDIR"
  source "$PYDIR/bin/activate"
  pip install -U pip
fi

# 3. JAX with CUDA 12 (for RTX 4090)
pip install -U "jax[cuda12]"

# 4. GraphCast and dependencies (from DeepMind GitHub)
pip install -U "https://github.com/deepmind/graphcast/archive/master.zip"

# 5. GCS and I/O
pip install -U google-cloud-storage xarray netCDF4 matplotlib pandas openpyxl

# 6. Verify JAX sees GPU
python3 -c "
import jax
devices = jax.devices()
print('JAX devices:', devices)
if not devices:
  print('WARNING: No GPU/TPU found.')
else:
  print('Backend:', devices[0].platform)
"

echo ""
echo "Setup done. Activate with: source $PYDIR/bin/activate"
echo "Then run: python run_graphcast_remote.py --lat 23.81 --lon 90.41 --outdir ./output"
