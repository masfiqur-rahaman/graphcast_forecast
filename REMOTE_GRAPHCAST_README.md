# Running GraphCast on a remote machine (SSH)

For a remote Ubuntu 22.04 machine with **RTX 4090 24GB**, **i9-13900**, **64GB RAM**.

## One-time setup on the remote machine

1. **SSH in**
   ```bash
   ssh user@remote-host
   ```

2. **NVIDIA driver (if not already)**  
   RTX 4090 needs driver 525+ (CUDA 12). Check:
   ```bash
   nvidia-smi
   ```
   If needed: `sudo apt update && sudo apt install -y nvidia-driver-535` (or newer), then reboot.

3. **Clone/copy this repo** (or at least these files) to the remote:
   - `run_graphcast_remote_setup.sh`
   - `run_graphcast_remote.py`

4. **Run setup once**
   ```bash
   chmod +x run_graphcast_remote_setup.sh
   ./run_graphcast_remote_setup.sh
   ```
   This creates a venv `./venv_graphcast`, installs JAX with CUDA 12, GraphCast, and dependencies.  
   To use a different directory: `PYDIR=~/envs/graphcast ./run_graphcast_remote_setup.sh`

5. **Activate the environment**
   ```bash
   source ./venv_graphcast/bin/activate   # or path you used for PYDIR
   ```

## Run a forecast (over SSH)

```bash
# Activate env (if not already)
source ./venv_graphcast/bin/activate

# Default location (Dhaka), small model, output in ./graphcast_output
python run_graphcast_remote.py --lat 23.81 --lon 90.41 --outdir ./graphcast_output

# Best model (0.25°, 37 levels) – needs more GPU memory
python run_graphcast_remote.py --lat 23.81 --lon 90.41 --model best --outdir ./graphcast_output

# Custom location
python run_graphcast_remote.py --lat 24.0 --lon 91.0 --outdir ./my_run
```

Outputs in `--outdir`:
- `graphcast_forecast_{lat}_{lon}.xlsx` – table of all variables
- `graphcast_forecast_{lat}_{lon}.png` – 5-panel plot

## Run in background (so you can disconnect SSH)

```bash
screen -S graphcast
source ./venv_graphcast/bin/activate
python run_graphcast_remote.py --lat 23.81 --lon 90.41 --outdir ./graphcast_output
# Detach: Ctrl+A then D
# Reattach later: screen -r graphcast
```

Or with `nohup`:
```bash
nohup python run_graphcast_remote.py --lat 23.81 --lon 90.41 --outdir ./graphcast_output > graphcast.log 2>&1 &
```

## Copy results back to your laptop

From your **local** machine:
```bash
scp user@remote-host:~/path/to/graphcast_output/*.xlsx ./
scp user@remote-host:~/path/to/graphcast_output/*.png ./
```

## Requirements on remote

- Ubuntu 22.04 (or similar Linux)
- Python 3.9+
- NVIDIA driver 525+ (for RTX 4090 / CUDA 12)
- Internet (to download model/data from GCS the first time)

Your hardware (4090 24GB, i9-13900, 64GB RAM) is more than enough for both `--model small` and `--model best`.
