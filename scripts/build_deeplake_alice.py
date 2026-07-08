import json
import os
import re
import shutil
from pathlib import Path

import cv2
import deeplake
import hydra
import numpy as np
import pandas as pd
import pyrootutils
from omegaconf import DictConfig
from scipy.io import loadmat


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

os.environ["HYDRA_FULL_ERROR"] = "1"

CHANNELS = ["direct_n", "diffuse", "temp", "rh", "pressure", "CSI"]
CONTEXT_CHANNELS = [f"ALICE_{idx}" for idx in range(6)]
STATION = "ALICE"
TIMESTAMP_PATTERN = re.compile(r"(\d{12})_s\.mat$")


def create_tensor(ds, name: str, data, info=None, dtype=None):
    tensor = ds.create_tensor(
        name,
        htype="generic",
        dtype=dtype or "unspecified",
        chunk_compression="lz4",
    )
    if info:
        tensor.info.update(info)
    tensor.append(data)
    return tensor


def parse_timestamp(path: Path) -> pd.Timestamp:
    match = TIMESTAMP_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse timestamp from {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H%M")


def list_frames(split_dir: Path) -> pd.DataFrame:
    # Alice *_s.mat files are the actual satellite image data. *_t.mat and *_o.mat
    # are auxiliary products and are intentionally excluded from this dataset.
    records = [(parse_timestamp(path), path) for path in split_dir.rglob("*_s.mat")]
    if not records:
        raise FileNotFoundError(f"No *_s.mat files under {split_dir}")
    frame_df = pd.DataFrame(records, columns=["time_utc", "path"]).sort_values("time_utc")
    frame_df = frame_df.drop_duplicates("time_utc", keep="first").reset_index(drop=True)
    return frame_df


def resize_frame(path: Path, image_size: int) -> np.ndarray:
    data = loadmat(path)["data"].astype("float32") / 255.0
    channels = [
        cv2.resize(data[..., idx], (image_size, image_size), interpolation=cv2.INTER_AREA)
        for idx in range(data.shape[-1])
    ]
    return np.stack(channels, axis=0).astype("float16")


def load_context(frame_df: pd.DataFrame, image_size: int) -> np.ndarray:
    context = np.empty((len(frame_df), len(CONTEXT_CHANNELS), image_size, image_size), dtype="float16")
    for idx, path in enumerate(frame_df["path"]):
        context[idx] = resize_frame(Path(path), image_size)
        if (idx + 1) % 1000 == 0:
            print(f"loaded frames {idx + 1}/{len(frame_df)}")
    return context


def make_tvl1():
    return cv2.optflow.DualTVL1OpticalFlow_create(
        tau=0.3,
        theta=0.5,
        nscales=3,
        warps=5,
        epsilon=0.01,
        innnerIterations=10,
        outerIterations=2,
        scaleStep=0.5,
        gamma=0.1,
        medianFiltering=5,
    )


def report_frame_gaps(split_name: str, times: pd.DatetimeIndex, expected_step: pd.Timedelta) -> None:
    deltas = times.to_series().diff().dropna()
    gaps = deltas[deltas != expected_step]
    if gaps.empty:
        print(f"{split_name}: all frame deltas are {expected_step}")
        return
    print(f"{split_name}: {len(gaps)} non-{expected_step} frame gaps")
    for timestamp, delta in gaps.head(10).items():
        previous = timestamp - delta
        print(f"  gap {previous} -> {timestamp}: {delta}")


def compute_optical_flow(context: np.ndarray, times: pd.DatetimeIndex, expected_step: pd.Timedelta) -> np.ndarray:
    n, c, h, w = context.shape
    out = np.zeros((n, c * 2, h, w), dtype="float32")
    frames = context.astype("float32")
    for idx in range(1, n):
        if times[idx] - times[idx - 1] != expected_step:
            continue
        for channel in range(c):
            flow = make_tvl1().calc(frames[idx - 1, channel], frames[idx, channel], None)
            out[idx, 2 * channel : 2 * channel + 2] = flow.transpose(2, 0, 1)
        if (idx + 1) % 1000 == 0:
            print(f"optflow frames {idx + 1}/{n}")
    return out.astype("float16")


def read_station(csv_path: Path) -> pd.DataFrame:
    source = pd.read_csv(csv_path)
    source.index = pd.to_datetime(source["Timestamp_UTC [DD/MM/YYYY hh:mm]"], dayfirst=True)
    source = source.sort_index()
    source = source[~source.index.duplicated(keep="first")]

    irradiance_cols = [
        "Irradiance_MB0 [W/m-2]",
        "Irradiance_MB1 [W/m-2]",
        "Irradiance_MB2 [W/m-2]",
    ]
    temp_cols = ["UnitTemp_MB0 [deg C]", "UnitTemp_MB1 [deg C]", "UnitTemp_MB2 [deg C]"]
    voltage_cols = ["Voltage_MB0 [V]", "Voltage_MB1 [V]", "Voltage_MB2 [V]"]

    station = pd.DataFrame(index=source.index)
    irradiance = source[irradiance_cols].mean(axis=1).clip(lower=0)
    daylight = irradiance[irradiance > 20]
    csi_denominator = max(
        float(daylight.quantile(0.995)) if len(daylight) else float(irradiance.quantile(0.995)),
        1.0,
    )
    station["direct_n"] = irradiance
    station["diffuse"] = source[irradiance_cols].std(axis=1).fillna(0.0)
    station["temp"] = source[temp_cols].mean(axis=1)
    station["rh"] = source[voltage_cols].mean(axis=1)
    station["pressure"] = source["Hour_Local"].astype("float32")
    station["CSI"] = (irradiance / csi_denominator).clip(lower=0, upper=1.5)
    return station[CHANNELS].interpolate(limit_direction="both").fillna(0.0).astype("float32")


def make_station_frame(source: pd.DataFrame, times: pd.DatetimeIndex, alignment: str) -> pd.DataFrame:
    missing = times.difference(source.index)
    if alignment == "exact":
        if len(missing):
            sample = ", ".join(str(ts) for ts in missing[:10])
            raise ValueError(
                f"Station CSV is missing {len(missing)} satellite UTC timestamps. "
                f"First missing values: {sample}"
            )
        return source.loc[times].astype("float32")
    if alignment == "interpolate":
        return source.reindex(times).interpolate(limit_direction="both").fillna(0.0).astype("float32")
    raise ValueError(f"Unsupported station_alignment={alignment!r}; use 'exact' or 'interpolate'")


def local_coordinates(alice_root: Path, image_size: int):
    local = loadmat(alice_root / "Alice_256" / "local.mat")["data"].astype("float32")
    lon_grid = cv2.resize(local[..., 0], (image_size, image_size), interpolation=cv2.INTER_AREA)
    lat_grid = cv2.resize(local[..., 1], (image_size, image_size), interpolation=cv2.INTER_AREA)
    lat = lat_grid[:, image_size // 2].astype("float32")
    lon = lon_grid[image_size // 2, :].astype("float32")
    elevation = ((lat_grid - lat_grid.mean()) + (lon_grid - lon_grid.mean())).astype("float32")
    return lat, lon, elevation


def write_split(ds, split_name, frame_df, context, optflow, station_df, lat, lon, elevation, cfg):
    prefix = split_name
    context_tensor = ds.create_tensor(
        f"{prefix}/context/data",
        htype="generic",
        dtype="float16",
        chunk_compression="lz4",
    )
    context_tensor.info.update({"context_channels": CONTEXT_CHANNELS})
    context_tensor.info.update({"source_file_pattern": "*_s.mat"})
    context_tensor.extend(context)
    create_tensor(ds, f"{prefix}/context/time_utc", frame_df["time_utc"].to_numpy(dtype="datetime64[ns]"))
    create_tensor(ds, f"{prefix}/context/latitude", lat.astype("float32"))
    create_tensor(ds, f"{prefix}/context/longitude", lon.astype("float32"))
    create_tensor(ds, f"{prefix}/context/elevation", elevation.astype("float32"))

    optflow_tensor = ds.create_tensor(
        f"{prefix}/ctx_opt_flow/data",
        htype="generic",
        dtype="float16",
        chunk_compression="lz4",
    )
    optflow_tensor.info.update(
        {
            "optflow_channels": [
                name
                for channel in CONTEXT_CHANNELS
                for name in (f"{channel}_vx", f"{channel}_vy")
            ]
        }
    )
    optflow_tensor.extend(optflow)

    station_tensor = ds.create_tensor(
        f"{prefix}/{STATION}/data",
        htype="generic",
        dtype="float32",
        chunk_compression="lz4",
    )
    station_tensor.info.update(
        {
            "timeseries_channels": CHANNELS,
            "coordinates": [float(cfg.station_lat), float(cfg.station_lon)],
            "elevation": float(cfg.station_elevation),
        }
    )
    station_tensor.append(station_df.to_numpy(dtype="float32"))
    create_tensor(ds, f"{prefix}/{STATION}/time_utc", frame_df["time_utc"].to_numpy(dtype="datetime64[ns]"))


@hydra.main(version_base="1.2", config_path=str(root / "configs"), config_name="build_deeplake_alice.yaml")
def main(cfg: DictConfig) -> None:
    alice_root = Path(cfg.alice_root)
    output_path = Path(cfg.output_path)
    stats_path = Path(cfg.stats_path)
    image_size = int(cfg.image_size)
    expected_step = pd.Timedelta(minutes=int(cfg.expected_step_minutes))
    if str(cfg.satellite_timestamp_timezone).lower() != "utc":
        raise ValueError("Alice satellite filenames are expected to be UTC timestamps")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    station_source = read_station(alice_root / "alice-springs_irradiancedata.csv")
    lat, lon, elevation = local_coordinates(alice_root, image_size)
    split_dirs = {
        "alice_train": alice_root / "Alice_256",
        "alice_val": alice_root / "Alice_val",
        "alice_test": alice_root / "Alice_TEST",
    }

    ds = deeplake.empty(str(output_path), overwrite=True)
    train_station_df = None
    for split_name, split_dir in split_dirs.items():
        frame_df = list_frames(split_dir)
        times = pd.DatetimeIndex(frame_df["time_utc"])
        report_frame_gaps(split_name, times, expected_step)
        station_df = make_station_frame(station_source, times, str(cfg.station_alignment))
        context = load_context(frame_df, image_size)
        optflow = compute_optical_flow(context, times, expected_step)
        write_split(ds, split_name, frame_df, context, optflow, station_df, lat, lon, elevation, cfg)
        ds.commit(f"write {split_name}")
        if split_name == "alice_train":
            train_station_df = station_df
        print(f"{split_name}: frames={len(frame_df)}, {times[0]}..{times[-1]}")

    stats = {
        STATION: {
            channel: {
                "mean": str(float(train_station_df[channel].mean())),
                "std": str(max(float(train_station_df[channel].std() or 1.0), 1.0)),
            }
            for channel in CHANNELS
        }
    }
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    print(f"Deep Lake Alice dataset: {output_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
