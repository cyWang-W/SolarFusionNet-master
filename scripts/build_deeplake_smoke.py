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

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

os.environ["HYDRA_FULL_ERROR"] = "1"


CHANNELS = ["direct_n", "diffuse", "temp", "rh", "pressure", "CSI"]


def read_bsrn_tab(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        lines = fp.readlines()

    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date/Time"))
    df = pd.read_csv(path, sep="\t", skiprows=header_idx, encoding="utf-8", encoding_errors="replace")
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df = df.set_index("Date/Time").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def make_station_frame(tab_path: Path, target_times: pd.DatetimeIndex) -> pd.DataFrame:
    source = read_bsrn_tab(tab_path)
    source_day = source.loc[source.index.normalize() == source.index[0].normalize()]
    source_5min = source_day.resample("5min").mean().interpolate(limit_direction="both")

    if len(source_5min) < len(target_times):
        raise ValueError(f"Not enough station rows: {len(source_5min)} < {len(target_times)}")

    source_5min = source_5min.iloc[: len(target_times)].copy()
    source_5min.index = target_times

    swd = source_5min.get("SWD [W/m**2]", pd.Series(0.0, index=target_times)).clip(lower=0)
    csi_denominator = max(float(swd.quantile(0.99)), 1.0)

    station = pd.DataFrame(index=target_times)
    station["direct_n"] = source_5min.get("DIR [W/m**2]", 0.0)
    station["diffuse"] = source_5min.get("DIF [W/m**2]", 0.0)
    station["temp"] = source_5min.get("T2 [°C]", source_5min.get("T2 [掳C]", 0.0))
    station["rh"] = source_5min.get("RH [%]", 0.0)
    station["pressure"] = source_5min.get("PoPoPoPo [hPa]", 0.0)
    station["CSI"] = (swd / csi_denominator).clip(lower=0, upper=1.5)
    return station[CHANNELS].fillna(0.0).astype("float32")


def create_tensor(ds, name: str, data, info=None):
    tensor = ds.create_tensor(name, htype="generic", chunk_compression="lz4", exist_ok=True)
    if info:
        tensor.info.update(info)
    tensor.append(data)
    return tensor


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="build_deeplake_smoke.yaml")
def main(cfg: DictConfig) -> None:
    zarr_path = Path(cfg.reproject_zarr)
    tab_path = Path(cfg.station_tab)
    output_path = Path(cfg.output_path)
    stats_path = Path(cfg.stats_path)
    year = str(cfg.year)
    station = str(cfg.station)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    zarr = xr.open_zarr(zarr_path)
    context = zarr["data"].astype("float32").values
    times = pd.DatetimeIndex(zarr["time_utc"].values)
    station_df = make_station_frame(tab_path, times)

    ds = deeplake.empty(str(output_path), overwrite=True)
    context_tensor = ds.create_tensor(f"{year}/context/data", htype="generic", dtype="float32", chunk_compression="lz4")
    context_tensor.info.update({"context_channels": list(map(str, zarr["channel"].values.tolist()))})
    context_tensor.extend(context)
    create_tensor(ds, f"{year}/context/time_utc", times.to_numpy(dtype="datetime64[ns]"))
    create_tensor(ds, f"{year}/context/latitude", zarr["latitude"].values.astype("float32"))
    create_tensor(ds, f"{year}/context/longitude", zarr["longitude"].values.astype("float32"))
    lat_grid, lon_grid = np.meshgrid(
        zarr["latitude"].values.astype("float32"),
        zarr["longitude"].values.astype("float32"),
        indexing="ij",
    )
    elevation = (lat_grid - lat_grid.mean()) + (lon_grid - lon_grid.mean())
    create_tensor(ds, f"{year}/context/elevation", elevation.astype("float32"))

    station_tensor = ds.create_tensor(f"{year}/{station}/data", htype="generic", dtype="float32", chunk_compression="lz4")
    station_tensor.info.update(
        {
            "timeseries_channels": CHANNELS,
            "coordinates": [float(cfg.station_lat), float(cfg.station_lon)],
            "elevation": float(cfg.station_elevation),
        }
    )
    station_tensor.append(station_df.to_numpy(dtype="float32"))
    create_tensor(ds, f"{year}/{station}/time_utc", times.to_numpy(dtype="datetime64[ns]"))

    stats = {
        station: {
            channel: {
                "mean": str(float(station_df[channel].mean())),
                "std": str(max(float(station_df[channel].std() or 1.0), 1.0)),
            }
            for channel in CHANNELS
        }
    }
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    print(f"Deep Lake smoke dataset: {output_path}")
    print(f"Stats: {stats_path}")
    print(f"context={context.shape}, station={station_df.shape}, time={times[0]}..{times[-1]}")


if __name__ == "__main__":
    main()
