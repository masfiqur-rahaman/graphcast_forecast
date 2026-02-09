#!/usr/bin/env python3
"""
Run GraphCast 10-day forecast at a single location (headless, for SSH/remote).
Uses same logic as GraphCast_DeepMind_Dhaka_10day.ipynb; writes Excel + plot files.

Usage:
  python run_graphcast_remote.py --lat 23.81 --lon 90.41 --outdir ./output
  python run_graphcast_remote.py --lat 23.81 --lon 90.41 --model best --outdir ./output

Requires: run_graphcast_remote_setup.sh to have been run (JAX+CUDA, graphcast, GCS).
"""

import argparse
import dataclasses
import functools
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

# GCS and graphcast (need to be installed)
from google.cloud import storage
from graphcast import autoregressive, casting, checkpoint, data_utils, graphcast, normalization, rollout
import haiku as hk
import jax

FORECAST_DAYS = 10
STEPS_PER_DAY = 4
TOTAL_STEPS_10D = FORECAST_DAYS * STEPS_PER_DAY
DIR_PREFIX = "graphcast/"


def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.replace(".nc", "").split("_"))


def data_valid_for_model(file_name, model_config, task_config):
    parts = parse_file_parts(file_name.removesuffix(".nc") if file_name.endswith(".nc") else file_name)
    try:
        res = float(parts.get("res", 0))
    except (ValueError, TypeError):
        res = 0
    try:
        levels = int(parts.get("levels", 0))
    except (ValueError, TypeError):
        levels = 0
    precip_ok = (
        ("total_precipitation_6hr" in task_config.input_variables and parts.get("source") in ("era5", "fake"))
        or ("total_precipitation_6hr" not in task_config.input_variables and parts.get("source") in ("hres", "fake"))
    )
    return (
        model_config.resolution in (0, res)
        and len(task_config.pressure_levels) == levels
        and precip_ok
    )


def steps_from_name(name):
    parts = parse_file_parts(name)
    try:
        return int(parts.get("steps", 0))
    except (ValueError, TypeError):
        return 0


def wet_bulb_stull(T_C, RH):
    a = np.arctan(0.151977 * (RH + 8.313659) ** 0.5)
    b = np.arctan(T_C + RH) - np.arctan(RH - 1.676331)
    c = 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035
    return T_C * a + b + c


def main():
    p = argparse.ArgumentParser(description="GraphCast 10-day forecast at (lat, lon), headless.")
    p.add_argument("--lat", type=float, default=23.81, help="Latitude (°N)")
    p.add_argument("--lon", type=float, default=90.41, help="Longitude (°E)")
    p.add_argument("--model", choices=("small", "best"), default="small", help="small (1deg/13lev) or best (0.25deg/37lev)")
    p.add_argument("--outdir", type=str, default="./graphcast_output", help="Output directory for Excel and plots")
    args = p.parse_args()

    LAT, LON = args.lat, args.lon
    os.makedirs(args.outdir, exist_ok=True)

    print("JAX devices:", jax.devices())
    print("Connecting to GCS dm_graphcast ...")
    gcs = storage.Client.create_anonymous_client()
    bucket = gcs.get_bucket("dm_graphcast")

    params_options = [b.name.removeprefix(DIR_PREFIX + "params/") for b in bucket.list_blobs(prefix=DIR_PREFIX + "params/") if b.name.removeprefix(DIR_PREFIX + "params/")]
    if args.model == "best":
        preferred = "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
        fallback = [x for x in params_options if "0.25" in x and "37" in x and "small" not in x.lower()]
    else:
        preferred = "GraphCast_small - ERA5 1979-2015 - resolution 1 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
        fallback = [x for x in params_options if "small" in x.lower() and "13" in x]
    params_file = preferred if preferred in params_options else (fallback[0] if fallback else params_options[0])
    print("Params:", params_file)

    print("Loading checkpoint ...")
    with bucket.blob(f"{DIR_PREFIX}params/{params_file}").open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params, state = ckpt.params, {}
    model_config, task_config = ckpt.model_config, ckpt.task_config

    dataset_options = [b.name.removeprefix(DIR_PREFIX + "dataset/") for b in bucket.list_blobs(prefix=DIR_PREFIX + "dataset/") if b.name.removeprefix(DIR_PREFIX + "dataset/")]
    valid = [f for f in dataset_options if data_valid_for_model(f, model_config, task_config)]
    if not valid:
        valid = dataset_options
    dataset_file = max(valid, key=steps_from_name)
    available_steps = steps_from_name(dataset_file)
    eval_steps = min(TOTAL_STEPS_10D, max(1, available_steps - 2))
    print("Dataset:", dataset_file, "| forecast steps:", eval_steps)

    print("Loading dataset ...")
    with bucket.blob(f"{DIR_PREFIX}dataset/{dataset_file}").open("rb") as f:
        example_batch = xarray.load_dataset(f).compute()
    assert example_batch.sizes["time"] >= 2 + eval_steps

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{eval_steps * 6}h"),
        **dataclasses.asdict(task_config),
    )

    print("Loading normalization stats ...")
    with bucket.blob(DIR_PREFIX + "stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with bucket.blob(DIR_PREFIX + "stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with bucket.blob(DIR_PREFIX + "stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()

    def construct_wrapped_graphcast(mconfig, tconfig):
        pred = graphcast.GraphCast(mconfig, tconfig)
        pred = casting.Bfloat16Cast(pred)
        pred = normalization.InputsAndResiduals(pred, diffs_stddev_by_level=diffs_stddev_by_level, mean_by_level=mean_by_level, stddev_by_level=stddev_by_level)
        return autoregressive.Predictor(pred, gradient_checkpointing=True)

    @hk.transform_with_state
    def run_forward(mconfig, tconfig, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(mconfig, tconfig)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    with_configs = lambda fn: functools.partial(fn, mconfig=model_config, tconfig=task_config)
    with_params = lambda fn: functools.partial(fn, params=params, state=state)
    drop_state = lambda fn: lambda **kw: fn(**kw)[0]
    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

    print("Running autoregressive rollout ...")
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )
    print("Rollout done.")

    lat_dim = "lat" if "lat" in predictions.coords else "latitude"
    lon_dim = "lon" if "lon" in predictions.coords else "longitude"
    point = predictions.sel({lat_dim: LAT, lon_dim: LON}, method="nearest")

    t0 = example_batch.time.isel(time=0).values
    try:
        base = pd.Timestamp(t0) + pd.Timedelta(hours=6)
    except Exception:
        base = pd.Timestamp("2000-01-01") + pd.Timedelta(hours=6)
    times = [base + pd.Timedelta(hours=6 * i) for i in range(eval_steps)]

    t2m_k = point["2m_temperature"].squeeze().values if "2m_temperature" in point else np.full(eval_steps, np.nan)
    air_temperature = (t2m_k - 273.15) if np.nanmin(t2m_k) > 200 else t2m_k

    if "specific_humidity" in point and "level" in point["specific_humidity"].dims:
        q_surf = point["specific_humidity"].isel(level=-1).squeeze().values
    else:
        q_surf = np.full(eval_steps, np.nan)
    p_surf = point["mean_sea_level_pressure"].squeeze().values if "mean_sea_level_pressure" in point else 101325.0
    p_surf = np.broadcast_to(np.atleast_1d(p_surf).flat[0] if np.isscalar(p_surf) else p_surf, eval_steps)
    e_sat = 611.2 * np.exp(17.67 * air_temperature / (air_temperature + 243.5))
    e = np.clip(q_surf * p_surf / (0.622 + 0.378 * np.clip(q_surf, 1e-10, 1)), 1e-10, None)
    relative_humidity = np.clip(100 * e / (e_sat + 1e-10), 0, 100)
    relative_humidity = np.where(np.isnan(q_surf), np.nan, relative_humidity)

    forc_pt = eval_forcings.sel({lat_dim: LAT, lon_dim: LON}, method="nearest")
    if "toa_incident_solar_radiation" in forc_pt:
        solar_radiation = np.asarray(forc_pt["toa_incident_solar_radiation"].squeeze().values)
    else:
        solar_radiation = np.full(eval_steps, np.nan)
    if np.size(solar_radiation) != eval_steps:
        solar_radiation = np.broadcast_to(np.nanmean(solar_radiation) if np.size(solar_radiation) else np.nan, eval_steps)

    u10 = point["10m_u_component_of_wind"].squeeze().values if "10m_u_component_of_wind" in point else np.zeros(eval_steps)
    v10 = point["10m_v_component_of_wind"].squeeze().values if "10m_v_component_of_wind" in point else np.zeros(eval_steps)
    wind_speed = np.sqrt(np.asarray(u10) ** 2 + np.asarray(v10) ** 2)

    Tw = wet_bulb_stull(air_temperature, relative_humidity)
    WBGT = 0.7 * Tw + 0.3 * air_temperature
    WBGT = np.where(np.isnan(air_temperature) | np.isnan(relative_humidity), np.nan, WBGT)

    df = pd.DataFrame({
        "datetime": times,
        "day": [t.strftime("%Y-%m-%d") for t in times],
        "hour": [t.hour for t in times],
        "air_temperature_2m_C": np.round(air_temperature, 2),
        "relative_humidity_pct": np.round(relative_humidity, 2),
        "solar_radiation_Wm2": np.round(solar_radiation, 2),
        "wind_speed_ms": np.round(wind_speed, 2),
        "WBGT_C": np.round(WBGT, 2),
    })

    excel_path = os.path.join(args.outdir, f"graphcast_forecast_{LAT}_{LON}.xlsx")
    df.to_excel(excel_path, index=False)
    print("Saved:", excel_path)

    x_labels = [t.strftime("%m-%d %H:%M") for t in times]
    x_ix = np.arange(len(times))
    step = max(1, len(x_ix) // 12)
    tick_ix = x_ix[::step]
    tick_lab = [x_labels[i] for i in range(0, len(x_labels), step)]

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"10-day forecast at {LAT}°N, {LON}°E (2 m)", fontsize=14)

    axes[0].plot(x_ix, df["air_temperature_2m_C"], "o-", color="C0", markersize=4)
    axes[0].set_ylabel("Air temperature (°C)")
    axes[0].set_title("1. Air temperature (2 m)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_ix, df["relative_humidity_pct"], "o-", color="C1", markersize=4)
    axes[1].set_ylabel("Relative humidity (%)")
    axes[1].set_title("2. Relative humidity")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_ix, df["solar_radiation_Wm2"], "o-", color="C2", markersize=4)
    axes[2].set_ylabel("Solar radiation (W/m²)")
    axes[2].set_title("3. Solar radiation (TOA)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(x_ix, df["wind_speed_ms"], "o-", color="C3", markersize=4)
    axes[3].set_ylabel("Wind speed (m/s)")
    axes[3].set_title("4. Wind speed (10 m)")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(x_ix, df["WBGT_C"], "o-", color="C4", markersize=4)
    axes[4].set_ylabel("WBGT (°C)")
    axes[4].set_xlabel("Time (day and hour)")
    axes[4].set_title("5. WBGT (Wet Bulb Globe Temperature)")
    axes[4].grid(True, alpha=0.3)

    plt.xticks(tick_ix, tick_lab, rotation=45, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, f"graphcast_forecast_{LAT}_{LON}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print("Saved:", plot_path)
    print("Done.")


if __name__ == "__main__":
    main()
