import json
import os
import shutil
from pathlib import Path

import deeplake
import hydra
import numpy as np
import pandas as pd
import pyrootutils
import xarray as xr
from omegaconf import DictConfig

from build_deeplake_smoke import CHANNELS, read_bsrn_tab

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

os.environ["HYDRA_FULL_ERROR"] = "1"


def read_station_year(tab_dir: Path) -> pd.DataFrame:
    frames = [read_bsrn_tab(path) for path in sorted(tab_dir.glob("IZA_radiation_2020-*.tab"))]
    if not frames:
        raise FileNotFoundError(f"No IZA_radiation_2020-*.tab files under {tab_dir}")
    source = pd.concat(frames).sort_index()
    source = source[~source.index.duplicated(keep="first")]
    source_5min = source.resample("5min").mean().interpolate(limit_direction="both")
    return source_5min


def make_station_frame(source_5min: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
    source_5min = source_5min.reindex(target_times).interpolate(limit_direction="both")
    swd = source_5min.get("SWD [W/m**2]", pd.Series(0.0, index=target_times)).clip(lower=0)
    daylight = swd[swd > 20]
    csi_denominator = max(float(daylight.quantile(0.99)) if len(daylight) else float(swd.quantile(0.99)), 1.0)

    station = pd.DataFrame(index=target_times)
    station["direct_n"] = source_5min.get("DIR [W/m**2]", 0.0)
    station["diffuse"] = source_5min.get("DIF [W/m**2]", 0.0)
    station["temp"] = source_5min.get("T2 [°C]", source_5min.get("T2 [掳C]", 0.0))
    station["rh"] = source_5min.get("RH [%]", 0.0)
    station["pressure"] = source_5min.get("PoPoPoPo [hPa]", 0.0)
    station["CSI"] = (swd / csi_denominator).clip(lower=0, upper=1.5)
    return station[CHANNELS].fillna(0.0).astype("float32")


def create_tensor(ds, name: str, data, info=None, dtype=None):
    tensor = ds.create_tensor(name, htype="generic", dtype=dtype or "unspecified", chunk_compression="lz4")
    if info:
        tensor.info.update(info)
    tensor.append(data)
    return tensor


def write_split(ds, split_name: str, context, times, lat, lon, elevation, station_df, station, cfg):
    prefix = split_name
    context_tensor = ds.create_tensor(
        f"{prefix}/context/data",
        htype="generic",
        dtype=str(context.dtype),
        chunk_compression="lz4",
    )
    context_tensor.info.update({"context_channels": ["HRV"]})
    context_tensor.extend(context)
    create_tensor(ds, f"{prefix}/context/time_utc", times.to_numpy(dtype="datetime64[ns]"))
    create_tensor(ds, f"{prefix}/context/latitude", lat.astype("float32"))
    create_tensor(ds, f"{prefix}/context/longitude", lon.astype("float32"))
    create_tensor(ds, f"{prefix}/context/elevation", elevation.astype("float32"))

    station_tensor = ds.create_tensor(f"{prefix}/{station}/data", htype="generic", dtype="float32", chunk_compression="lz4")
    station_tensor.info.update(
        {
            "timeseries_channels": CHANNELS,
            "coordinates": [float(cfg.station_lat), float(cfg.station_lon)],
            "elevation": float(cfg.station_elevation),
        }
    )
    station_tensor.append(station_df.to_numpy(dtype="float32"))
    create_tensor(ds, f"{prefix}/{station}/time_utc", times.to_numpy(dtype="datetime64[ns]"))


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="build_deeplake_year.yaml")
def main(cfg: DictConfig) -> None:
    reproject_dir = Path(cfg.reproject_dir)
    station_dir = Path(cfg.station_dir)
    output_path = Path(cfg.output_path)
    stats_path = Path(cfg.stats_path)
    station = str(cfg.station)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    zarr_paths = sorted(reproject_dir.glob("2020-*.zarr"))
    if not zarr_paths:
        raise FileNotFoundError(f"No daily zarrs under {reproject_dir}")
    datasets = [xr.open_zarr(path) for path in zarr_paths]
    zarr = xr.concat(datasets, dim="time_utc").sortby("time_utc")
    _, unique_idx = np.unique(zarr["time_utc"].values, return_index=True)
    zarr = zarr.isel(time_utc=np.sort(unique_idx))

    times = pd.DatetimeIndex(zarr["time_utc"].values)
    station_source = read_station_year(station_dir)
    station_df = make_station_frame(station_source, times)
    context = zarr["data"].astype("float16").values
    lat = zarr["latitude"].values.astype("float32")
    lon = zarr["longitude"].values.astype("float32")
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    elevation = (lat_grid - lat_grid.mean()) + (lon_grid - lon_grid.mean())

    split_masks = {
        "2020_train": times < pd.Timestamp("2020-11-01"),
        "2020_val": (times >= pd.Timestamp("2020-11-01")) & (times < pd.Timestamp("2020-12-01")),
        "2020_test": times >= pd.Timestamp("2020-12-01"),
    }

    ds = deeplake.empty(str(output_path), overwrite=True)
    for split_name, mask in split_masks.items():
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ValueError(f"Split {split_name} is empty")
        write_split(
            ds,
            split_name,
            context[idx],
            times[idx],
            lat,
            lon,
            elevation,
            station_df.iloc[idx],
            station,
            cfg,
        )
        print(f"{split_name}: frames={len(idx)}, {times[idx][0]}..{times[idx][-1]}")

    train_df = station_df.loc[times < pd.Timestamp("2020-11-01")]
    stats = {
        station: {
            channel: {
                "mean": str(float(train_df[channel].mean())),
                "std": str(max(float(train_df[channel].std() or 1.0), 1.0)),
            }
            for channel in CHANNELS
        }
    }
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    print(f"Deep Lake year dataset: {output_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
