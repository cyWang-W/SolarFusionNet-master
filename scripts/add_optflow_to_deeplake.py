from pathlib import Path
from typing import Optional, Tuple

import cv2
import deeplake
import numpy as np


DATASET_PATH = Path(r"E:\solarfusionnet_preprocess\deeplake_iza_2020")
SPLITS = ["2020_train", "2020_val", "2020_test"]


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


def compute_batch(
    context_tensor, start: int, end: int, previous_frame: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    frames = context_tensor[start:end, 0].numpy().astype("float32")
    out = np.zeros((len(frames), 2, frames.shape[-2], frames.shape[-1]), dtype="float32")
    first_idx = 0
    if previous_frame is not None and len(frames):
        optical_flow = make_tvl1()
        flow = optical_flow.calc(previous_frame, frames[0], None)
        out[0] = flow.transpose(2, 0, 1)
        first_idx = 1
    for local_idx in range(first_idx, len(frames)):
        if local_idx == 0:
            continue
        optical_flow = make_tvl1()
        flow = optical_flow.calc(frames[local_idx - 1], frames[local_idx], None)
        out[local_idx] = flow.transpose(2, 0, 1)
    return out, frames[-1]


def main() -> None:
    ds = deeplake.load(str(DATASET_PATH), read_only=False)
    for split in SPLITS:
        tensor_name = f"{split}/ctx_opt_flow/data"
        ctx = ds[f"{split}/context/data"]
        if tensor_name in ds.tensors:
            flow_tensor = ds[tensor_name]
            print(f"exists {tensor_name}, current shape={flow_tensor.shape}")
        else:
            flow_tensor = ds.create_tensor(
                tensor_name,
                htype="generic",
                dtype="float32",
                chunk_compression="lz4",
            )
            flow_tensor.info.update({"optflow_channels": ["HRV_vx", "HRV_vy"]})
        n = ctx.shape[0]
        batch_size = 512
        existing = flow_tensor.shape[0]
        if existing >= n:
            print(f"complete {tensor_name}, skipping")
            continue
        previous_frame = ctx[existing - 1, 0].numpy().astype("float32") if existing > 0 else None
        for start in range(existing, n, batch_size):
            end = min(start + batch_size, n)
            flows, previous_frame = compute_batch(ctx, start, end, previous_frame)
            flow_tensor.extend(flows)
            print(f"{split}: {end}/{n}")
        ds.commit(f"add optical flow for {split}")


if __name__ == "__main__":
    main()
