"""
AdaBN: adapt a trained model to the held-out hospital without its ages and without retraining.

The BatchNorm layers normalize their inputs with mean / variance statistics estimated on the
training hospitals. AdaBN (Li et al., 2016) re-estimates these statistics on the images of the
held-out hospital (images only, no labels) and keeps every learned weight unchanged.

Protocol: the statistics are re-estimated on all out-of-site images (transductive setting: the
same images are then predicted, but their ages are never used). The in-site group is predicted
with the original model, since it comes from the training hospitals.

Outputs in the run folder:
  predictions_adabn_<checkpoint>.tsv / .png   same format as predict_ages.py

Run:  python scripts/adabn.py --data t1linear --test-site IOP [--augment]
"""

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_ages import IDS, load_model, model_path, plot, predict, summarize
from train_ixi_site_shift import AgeRegressor, add_run_args, age_table_path, make_dataset, run_dir_path


@torch.no_grad()
def adapt_batchnorm(model: AgeRegressor, data: pd.DataFrame, device: torch.device, images: str,
                    batch_size: int = 8) -> AgeRegressor:
    """Copy of the model whose BatchNorm statistics are re-estimated on the given images."""
    adapted = copy.deepcopy(model)
    bn_layers = [m for m in adapted.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bn_layers:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over all batches instead of an exponential one
    adapted.train()  # BatchNorm updates its running statistics in train mode (no weight is updated)

    # ClinicaDL requires the label column: filled with zeros, the true ages are never passed here
    dataset = make_dataset(data[IDS].reset_index(drop=True).assign(AgeMRI=0.0), data=images)
    dataset.eval()  # no augmentation
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i].image for i in range(start, min(start + batch_size, len(dataset)))]
        x = torch.stack([(b.data if hasattr(b, "data") else b).float() for b in batch]).to(device)
        adapted(x)
    print(f"  {len(bn_layers)} BatchNorm layers re-estimated on {len(dataset)} images")
    return adapted.eval()


def main():
    parser = argparse.ArgumentParser()
    add_run_args(parser)  # same options as for training, to find the run folder
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default="best-mae")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size used to re-estimate the statistics")
    args = parser.parse_args()

    run_dir = args.run_dir or run_dir_path(args)
    model_pt = model_path(run_dir, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_pt}  ({device})")
    model = load_model(model_pt, device)

    table = pd.read_csv(age_table_path(args.data), sep="\t")
    insite = table.merge(pd.read_csv(run_dir / "split" / "test_baseline.tsv", sep="\t")[IDS], on=IDS)
    outofsite = table.merge(pd.read_csv(run_dir / "out_of_site.tsv", sep="\t")[IDS], on=IDS)

    print(f"\nAdaBN on the {args.test_site} images (ages not used)")
    adapted = adapt_batchnorm(model, outofsite[IDS], device, args.data, args.batch_size)

    results = []
    for group, data, net in [("insite", insite, model), ("outofsite", outofsite, model),
                             ("outofsite_adabn", outofsite, adapted)]:
        print(f"\nPrediction {group} ({len(data)} images)")
        data = data.copy()
        data["predicted"] = predict(net, data, device, args.data)
        data["group"] = group
        results.append(data)

    df = pd.concat(results, ignore_index=True)
    df["error"] = df.predicted - df.AgeMRI
    df = df[IDS + ["site", "group", "AgeMRI", "predicted", "error"]]
    out_tsv = run_dir / f"predictions_adabn_{args.checkpoint}.tsv"
    df.round(2).to_csv(out_tsv, sep="\t", index=False)
    print(f"\n-> {out_tsv}")

    summarize(df)
    plot(df[df.group != "outofsite"], f"{run_dir.name} - AdaBN - {args.checkpoint}",
         run_dir / f"predictions_adabn_{args.checkpoint}.png")


if __name__ == "__main__":
    main()
