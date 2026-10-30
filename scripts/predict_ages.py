"""
Signed predictions (predicted - true age) of a model trained by train_ixi_site_shift.py.

ClinicaDL 2.0 only saves the absolute error at test time: this script reloads the checkpoint
and computes the predicted age of every subject of the insite (internal test) and outofsite groups.

Outputs in the run folder:
  predictions_<checkpoint>.tsv   participant_id, session_id, site, group, AgeMRI, predicted, error
  predictions_<checkpoint>.png   predicted vs true age, and error vs true age
  validation_curve.png           validation MAE per epoch

Run:  python scripts/predict_ages.py --test-site IOP [--data t1linear --augment --scheduler plateau --seed <n>]
                                                  [--checkpoint final]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ixi_site_shift import AgeRegressor, add_run_args, age_table_path, make_dataset, run_dir_path

AGE_BINS = [0, 35, 50, 65, 100]
IDS = ["participant_id", "session_id"]


def model_path(run_dir: Path, checkpoint: str) -> Path:
    """best-mae -> best_models/best-mae, epoch-<n> -> checkpoints/epoch-<n>, final -> final."""
    models_dir = run_dir / "maps" / "training" / "split-0" / "models"
    return (models_dir / "best_models" / checkpoint if checkpoint.startswith("best")
            else models_dir / "checkpoints" / checkpoint if checkpoint.startswith("epoch")
            else models_dir / checkpoint) / "model.pt"


def load_model(model_pt: Path, device: torch.device) -> AgeRegressor:
    state = torch.load(model_pt, map_location="cpu")
    state = {k.removeprefix("network."): v for k, v in state.items()}
    if "age_mean" not in state:  # first model (v1a): a DenseNet that outputs the age directly
        state = {f"net.{k}": v for k, v in state.items()}
        state |= {"age_mean": torch.tensor(0.0), "age_std": torch.tensor(1.0)}
    model = AgeRegressor()
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict(model: AgeRegressor, data: pd.DataFrame, device: torch.device, images: str = "bids2mm") -> list[float]:
    dataset = make_dataset(data, data=images)
    dataset.eval()  # never augment at prediction time
    preds = []
    for i in range(len(dataset)):
        img_obj = dataset[i].image
        tensor = img_obj.data if hasattr(img_obj, "data") else img_obj
        preds.append(model(tensor.float().unsqueeze(0).to(device)).item())
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(dataset)}")
    return preds


def summarize(df: pd.DataFrame) -> None:
    for group, d in df.groupby("group", sort=False):
        r = d.AgeMRI.corr(d.predicted)
        print(f"\n{group} : n={len(d)}  MAE={d.error.abs().mean():.2f}  "
              f"mean bias={d.error.mean():+.2f} years  r={r:.2f}")
        by_age = d.groupby(pd.cut(d.AgeMRI, AGE_BINS), observed=True)["error"]
        table = pd.DataFrame({"n": by_age.size(), "MAE": by_age.apply(lambda e: e.abs().mean()),
                              "bias": by_age.mean()})
        print(table.round(2).to_string())


def plot(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for group, d in df.groupby("group", sort=False):
        ax1.scatter(d.AgeMRI, d.predicted, s=15, alpha=0.7, label=f"{group} (n={len(d)})")
        ax2.scatter(d.AgeMRI, d.error, s=15, alpha=0.7, label=group)
    values = df[["AgeMRI", "predicted"]]
    lims = [min(15, values.min().min() - 2), max(90, values.max().max() + 2)]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set(xlabel="True age", ylabel="Predicted age", xlim=lims, ylim=lims, title="Predicted vs true age")
    ax2.axhline(0, color="k", lw=1, ls="--")
    ax2.set(xlabel="True age", ylabel="Predicted - true (years)", title="Signed error")
    ax1.legend()
    fig.suptitle(title)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"-> {path}")


def plot_validation_curve(run_dir: Path) -> None:
    tsv = run_dir / "maps" / "training" / "split-0" / "validation_metrics" / "aggregated.tsv"
    if not tsv.exists():
        return
    val = pd.read_csv(tsv, sep="\t")
    ax = val.plot(x=val.columns[0], y="mae", marker="o", legend=False, title="Validation MAE per epoch")
    ax.set_ylabel("MAE (years)")
    ax.figure.savefig(run_dir / "validation_curve.png", bbox_inches="tight", dpi=120)
    plt.close(ax.figure)
    print(f"-> {run_dir / 'validation_curve.png'}")


def main():
    parser = argparse.ArgumentParser()
    add_run_args(parser)  # same options as for training, to find the run folder
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="run folder (default: derived from the options above)")
    parser.add_argument("--checkpoint", default="best-mae", help="best-mae, final or epoch-<n>")
    args = parser.parse_args()

    run_dir = args.run_dir or run_dir_path(args)
    model_pt = model_path(run_dir, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_pt}  ({device})")
    model = load_model(model_pt, device)

    table = pd.read_csv(age_table_path(args.data), sep="\t")
    groups = {
        "insite": pd.read_csv(run_dir / "split" / "test_baseline.tsv", sep="\t")[IDS],
        "outofsite": pd.read_csv(run_dir / "out_of_site.tsv", sep="\t")[IDS],
    }
    results = []
    for group, ids in groups.items():
        data = table.merge(ids, on=IDS)
        print(f"\nPrediction {group} ({len(data)} images)")
        data["predicted"] = predict(model, data, device, args.data)
        data["group"] = group
        results.append(data)

    df = pd.concat(results, ignore_index=True)
    df["error"] = df.predicted - df.AgeMRI
    df = df[IDS + ["site", "group", "AgeMRI", "predicted", "error"]]
    out_tsv = run_dir / f"predictions_{args.checkpoint}.tsv"
    df.round(2).to_csv(out_tsv, sep="\t", index=False)
    print(f"\n-> {out_tsv}")

    summarize(df)
    plot(df, f"{run_dir.name} - {args.checkpoint}", run_dir / f"predictions_{args.checkpoint}.png")
    plot_validation_curve(run_dir)


if __name__ == "__main__":
    main()
