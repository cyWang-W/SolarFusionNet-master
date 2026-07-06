from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pred_dir = root / "None" / "train" / "runs" / "joty" / "predictions"
    pred = np.load(pred_dir / "all_predictions.npy").squeeze(-1)
    gt = np.load(pred_dir / "all_ground_truths.npy").squeeze(-1)

    out_dir = pred_dir
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    mae = float(np.mean(np.abs(pred - gt)))

    plt.figure(figsize=(12, 7))
    for idx in [0, 10, 25, 50, 75]:
        if idx >= len(pred):
            continue
        x = np.arange(pred.shape[1]) + idx * pred.shape[1]
        plt.plot(x, gt[idx], color="black", alpha=0.55, linewidth=1.5)
        plt.plot(x, pred[idx], alpha=0.8, linewidth=1.4)
    plt.title(f"IZA 2020 Test Forecast Samples | RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.xlabel("Forecast step blocks")
    plt.ylabel("CSI")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "forecast_samples.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(gt.ravel(), pred.ravel(), s=8, alpha=0.28)
    lo = float(min(gt.min(), pred.min()))
    hi = float(max(gt.max(), pred.max()))
    plt.plot([lo, hi], [lo, hi], color="red", linewidth=1.5)
    plt.title(f"Predicted vs Ground Truth | RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.xlabel("Ground truth CSI")
    plt.ylabel("Predicted CSI")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_scatter.png", dpi=180)
    plt.close()

    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as fp:
        fp.write(f"rmse={rmse}\nmae={mae}\n")
        fp.write(f"pred_shape={pred.shape}\nground_truth_shape={gt.shape}\n")
        fp.write(f"pred_min={float(pred.min())}\npred_max={float(pred.max())}\n")
        fp.write(f"ground_truth_min={float(gt.min())}\nground_truth_max={float(gt.max())}\n")

    print(out_dir / "forecast_samples.png")
    print(out_dir / "prediction_scatter.png")
    print(out_dir / "metrics.txt")


if __name__ == "__main__":
    main()
