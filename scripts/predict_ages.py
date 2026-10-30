"""
Prédictions signées (âge prédit - âge réel) du modèle entraîné par train_ixi_site_shift.py.

ClinicaDL 2.0 n'enregistre que l'erreur absolue lors du test : ce script recharge le checkpoint
et calcule l'âge prédit de chaque sujet des groupes insite (test interne) et outofsite.

Sorties dans le dossier du run :
  predictions_<checkpoint>.tsv   participant_id, session_id, site, group, AgeMRI, predicted, error
  predictions_<checkpoint>.png   âge prédit vs réel, et erreur vs âge réel
  validation_curve.png           MAE de validation par epoch

Lancer :  python scripts/predict_ages.py --test-site IOP
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


def load_model(model_pt: Path, device: torch.device) -> AgeRegressor:
    state = torch.load(model_pt, map_location="cpu")
    state = {k.removeprefix("network."): v for k, v in state.items()}
    if "age_mean" not in state:  # premier modèle (v1) : DenseNet qui sort directement l'âge
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
              f"biais moyen={d.error.mean():+.2f} ans  r={r:.2f}")
        by_age = d.groupby(pd.cut(d.AgeMRI, AGE_BINS), observed=True)["error"]
        table = pd.DataFrame({"n": by_age.size(), "MAE": by_age.apply(lambda e: e.abs().mean()),
                              "biais": by_age.mean()})
        print(table.round(2).to_string())


def plot(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for group, d in df.groupby("group", sort=False):
        ax1.scatter(d.AgeMRI, d.predicted, s=15, alpha=0.7, label=f"{group} (n={len(d)})")
        ax2.scatter(d.AgeMRI, d.error, s=15, alpha=0.7, label=group)
    values = df[["AgeMRI", "predicted"]]
    lims = [min(15, values.min().min() - 2), max(90, values.max().max() + 2)]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set(xlabel="Âge réel", ylabel="Âge prédit", xlim=lims, ylim=lims, title="Âge prédit vs réel")
    ax2.axhline(0, color="k", lw=1, ls="--")
    ax2.set(xlabel="Âge réel", ylabel="Prédit - réel (ans)", title="Erreur signée")
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
    ax = val.plot(x=val.columns[0], y="mae", marker="o", legend=False, title="MAE de validation par epoch")
    ax.set_ylabel("MAE (ans)")
    ax.figure.savefig(run_dir / "validation_curve.png", bbox_inches="tight", dpi=120)
    plt.close(ax.figure)
    print(f"-> {run_dir / 'validation_curve.png'}")


def main():
    parser = argparse.ArgumentParser()
    add_run_args(parser)  # same options as for training, to find the run folder
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="dossier du run (défaut : déduit des options ci-dessus)")
    parser.add_argument("--checkpoint", default="best-mae", help="best-mae, final ou epoch-<n>")
    args = parser.parse_args()

    run_dir = args.run_dir or run_dir_path(args)
    models_dir = run_dir / "maps" / "training" / "split-0" / "models"
    model_pt = (models_dir / "best_models" / args.checkpoint if args.checkpoint.startswith("best")
                else models_dir / "checkpoints" / args.checkpoint if args.checkpoint.startswith("epoch")
                else models_dir / args.checkpoint) / "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modèle : {model_pt}  ({device})")
    model = load_model(model_pt, device)

    table = pd.read_csv(age_table_path(args.data), sep="\t")
    groups = {
        "insite": pd.read_csv(run_dir / "split" / "test_baseline.tsv", sep="\t")[IDS],
        "outofsite": pd.read_csv(run_dir / "out_of_site.tsv", sep="\t")[IDS],
    }
    results = []
    for group, ids in groups.items():
        data = table.merge(ids, on=IDS)
        print(f"\nPrédiction {group} ({len(data)} images)")
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
