from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_id = os.environ.get("RUN_ID", "alice")
    pred_dir = root / "outputs" / "train" / "runs" / run_id / "predictions"
    pred = np.load(pred_dir / "all_predictions.npy").squeeze(-1)
    gt = np.load(pred_dir / "all_ground_truths.npy").squeeze(-1)
    time_coords = np.load(pred_dir / "all_time_coords.npy")

    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    mae = float(np.mean(np.abs(pred - gt)))
    horizon_steps = pred.shape[1]
    horizon_hours = np.arange(1, horizon_steps + 1) * 0.25

    if horizon_steps == 1:
        x = np.arange(len(pred))
        plt.figure(figsize=(13, 6))
        plt.plot(x, gt[:, 0], color="#202124", linewidth=2.0, label="Ground truth")
        plt.plot(x, pred[:, 0], color="#d62728", linewidth=1.8, label="Prediction")
        plt.fill_between(x, gt[:, 0], pred[:, 0], color="#d62728", alpha=0.12, linewidth=0)
        plt.title(f"Alice Springs 15-minute CSI Forecast | Test RMSE={rmse:.3f}, MAE={mae:.3f}")
        plt.xlabel("Test sample index")
        plt.ylabel("CSI")
        plt.grid(alpha=0.22)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(pred_dir / "single_step_series.png", dpi=180)
        plt.savefig(pred_dir / "forecast_samples.png", dpi=180)
        plt.close()
    else:
        daylight = gt.mean(axis=1)
        candidate_order = np.argsort(daylight)[::-1]
        selected = []
        for idx in candidate_order:
            if len(selected) == 6:
                break
            if daylight[idx] < 0.1:
                continue
            if all(abs(int(idx) - int(prev)) >= 8 for prev in selected):
                selected.append(int(idx))
        if len(selected) < 6:
            selected = list(candidate_order[:6])
        selected = sorted(selected)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
        for ax, idx in zip(axes.ravel(), selected):
            sample_rmse = float(np.sqrt(np.mean((pred[idx] - gt[idx]) ** 2)))
            input_month, input_day, input_hour, input_minute = time_coords[idx, -1, :, 0, 0].astype(int)
            title = f"{input_month:02d}-{input_day:02d} {input_hour:02d}:{input_minute:02d} UTC | RMSE {sample_rmse:.3f}"
            ax.plot(horizon_hours, gt[idx], color="#202124", linewidth=2.4, label="Ground truth")
            ax.plot(horizon_hours, pred[idx], color="#d62728", linewidth=2.2, label="Prediction")
            ax.fill_between(horizon_hours, gt[idx], pred[idx], color="#d62728", alpha=0.12, linewidth=0)
            ax.set_title(title, fontsize=11)
            ax.set_xlim(float(horizon_hours[0]), float(horizon_hours[-1]))
            ax.set_ylim(-0.05, max(1.15, float(gt[selected].max()) + 0.05, float(pred[selected].max()) + 0.05))
            ax.grid(alpha=0.22)
        axes[0, 0].set_ylabel("CSI")
        axes[1, 0].set_ylabel("CSI")
        for ax in axes[1]:
            ax.set_xlabel("Forecast horizon (hours)")
        axes[0, 0].legend(loc="upper right", frameon=False)
        fig.suptitle(f"Alice Springs {horizon_hours[-1]:.1f}-hour CSI Forecasts | Test RMSE={rmse:.3f}, MAE={mae:.3f}", fontsize=15, y=0.98)
        plt.tight_layout()
        plt.savefig(pred_dir / "forecast_samples.png", dpi=180)
        plt.savefig(pred_dir / "forecast_panels.png", dpi=180)
        plt.close()

    plt.figure(figsize=(7, 6.5))
    plt.scatter(gt.ravel(), pred.ravel(), s=9, alpha=0.22, color="#1f77b4", edgecolors="none")
    lo = float(min(gt.min(), pred.min()))
    hi = float(max(gt.max(), pred.max()))
    plt.plot([lo, hi], [lo, hi], color="#d62728", linewidth=1.7)
    plt.title(f"Alice Predicted vs Ground Truth | RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.xlabel("Ground truth CSI")
    plt.ylabel("Predicted CSI")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(pred_dir / "prediction_scatter.png", dpi=180)
    plt.close()

    with (pred_dir / "metrics.txt").open("w", encoding="utf-8") as fp:
        fp.write(f"rmse={rmse}\nmae={mae}\n")
        fp.write(f"pred_shape={pred.shape}\nground_truth_shape={gt.shape}\n")
        fp.write(f"pred_min={float(pred.min())}\npred_max={float(pred.max())}\n")
        fp.write(f"ground_truth_min={float(gt.min())}\nground_truth_max={float(gt.max())}\n")
        fp.write(f"forecast_horizon_hours={float(horizon_hours[-1])}\n")
        fp.write("forecast_step_minutes=15\n")
        if horizon_steps > 1:
            fp.write(f"selected_samples={selected}\n")

    print(pred_dir / "forecast_samples.png")
    if horizon_steps == 1:
        print(pred_dir / "single_step_series.png")
    else:
        print(pred_dir / "forecast_panels.png")
    print(pred_dir / "prediction_scatter.png")
    print(pred_dir / "metrics.txt")


if __name__ == "__main__":
    main()
