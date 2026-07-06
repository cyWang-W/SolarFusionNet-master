from pathlib import Path
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import open_dict
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.satellite_ts import TSContextDataset

RUN_DIR = ROOT / "None" / "train" / "runs" / "joty"
OUT_DIR = RUN_DIR / "predictions"


def find_checkpoint() -> Path:
    checkpoints = sorted(
        (RUN_DIR / "checkpoints").glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    best = [path for path in checkpoints if path.name.startswith("epoch_")]
    candidates = best or checkpoints
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {RUN_DIR / 'checkpoints'}")
    return candidates[0]


def main() -> None:
    with hydra.initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.2"):
        cfg = hydra.compose(
            config_name="train.yaml",
            overrides=["experiment=iza_2020_converge", "extras.print_config=False"],
        )

    with open_dict(cfg):
        cfg.pl_module.model.batch_size = 1

    model = hydra.utils.instantiate(cfg.pl_module)
    ckpt = find_checkpoint()
    print(f"Using checkpoint: {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.cuda().eval()

    dataset = TSContextDataset(**cfg.datamodule.dataset, mode="test")

    scored = []
    for idx in range(0, len(dataset), 12):
        item = dataset[idx]
        mean_csi = float(item["target"].mean())
        if mean_csi > 0.15:
            scored.append((mean_csi, idx))
    selected = [idx for _, idx in sorted(scored, reverse=True)[:6]]
    selected = sorted(selected)

    preds = []
    targets = []
    labels = []
    time_axis = dataset.deeplake_ds["2020_test/IZA/time_utc"].numpy()[0]
    for idx in selected:
        batch = default_collate([dataset[idx]])
        batch = {
            key: (value.cuda() if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        with torch.no_grad():
            x_ts, x_ctx, y_ts, _, ctx_coords, ts_coords, time_coords = model.prepare_batch(batch)
            optical_flow = batch.get("optical_flow")
            if optical_flow is not None:
                optical_flow = optical_flow.float()
            y_hat, *_ = model(
                x_ctx,
                ctx_coords,
                x_ts,
                ts_coords,
                time_coords,
                mask=False,
                optical_flow=optical_flow,
            )
        preds.append(y_hat.squeeze().detach().cpu().numpy())
        targets.append(y_ts.squeeze().detach().cpu().numpy())
        _, _, step_idx_station, _ = dataset.year_mapping[idx]
        target_start = step_idx_station + dataset.seq_len
        labels.append(str(time_axis[target_start])[:16])

    preds = np.asarray(preds)
    targets = np.asarray(targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))

    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 12), sharex=True, sharey=True)
    for ax, pred, target, label in zip(axes, preds, targets, labels):
        ax.plot(target, color="black", linewidth=2, label="Ground truth")
        ax.plot(pred, color="#1f77b4", linewidth=2, label="Prediction")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.set_ylabel("CSI")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Forecast step, 5 min")
    fig.suptitle(f"IZA 2020 Daylight Test Forecasts | RMSE={rmse:.3f}, MAE={mae:.3f}", y=0.995)
    fig.tight_layout()
    out = OUT_DIR / "daylight_forecast_samples.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print(out)
    print(f"selected={selected}")
    print(f"rmse={rmse} mae={mae}")


if __name__ == "__main__":
    main()
