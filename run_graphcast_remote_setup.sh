#!/usr/bin/env bash
# Setup for running GraphCast (DeepMind) on remote Ubuntu 22.04 with RTX 4090.
# Uses conda. Run once on the remote machine, then use run_graphcast_remote.py

set -e

CONDA_ENV="${CONDA_ENV:-graphcast}"
echo "=== GraphCast remote setup (Conda, Ubuntu 22.04, RTX 4090) ==="

# 1. Check NVIDIA driver (need 525+ for CUDA 12 / RTX 4090)
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. Install NVIDIA driver (e.g. sudo apt install nvidia-driver-535)."
fi

# 2. Ensure conda is available
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# 3. Create conda env (or use existing)
if conda env list | grep -q "^${CONDA_ENV} "; then
  echo "Conda env '$CONDA_ENV' already exists. Activating and updating..."
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "Creating conda env '$CONDA_ENV' with Python 3.11..."
  eval "$(conda shell.bash hook)"
  conda create -n "$CONDA_ENV" python=3.11 -y
  conda activate "$CONDA_ENV"
fi

pip install -U pip

# 4. JAX with CUDA 12 (for RTX 4090)
pip install -U "jax[cuda12]"

# 5. GraphCast and dependencies (from DeepMind GitHub)
pip install -U "https://github.com/deepmind/graphcast/archive/master.zip"

# 6. GCS and I/O
pip install -U google-cloud-storage xarray netCDF4 matplotlib pandas openpyxl

# 7. Verify JAX sees GPU
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
echo "Setup done. Activate with: conda activate $CONDA_ENV"
echo "Then run: python run_graphcast_remote.py --lat 23.81 --lon 90.41 --outdir ./output"
