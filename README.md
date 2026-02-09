# GraphCast 10-day forecast (Dhaka / any location)

GraphCast-based 10-day weather forecast at 2 m: air temperature, relative humidity, solar radiation, wind speed, WBGT. Outputs Excel + plots.

- **Notebook:** `GraphCast_DeepMind_Dhaka_10day.ipynb` — run in Colab or Jupyter (set location and model in the notebook).
- **Remote/SSH:** use `run_graphcast_remote_setup.sh` once, then `run_graphcast_remote.py` for headless runs.

See **REMOTE_GRAPHCAST_README.md** for setup and usage on a remote machine (e.g. Ubuntu + RTX 4090 over SSH).

## Quick start (remote)

Requires Conda. Then:

```bash
./run_graphcast_remote_setup.sh
conda activate graphcast
python run_graphcast_remote.py --lat 23.81 --lon 90.41 --outdir ./graphcast_output
```
