"""
DANN (domain-adversarial neural network, Ganin et al., JMLR 2016) for brain age, starting from a
model trained by train_ixi_site_shift.py.

Two heads share the DenseNet features f (1024 values per image):
  - the age head (the original last layer) is trained on the training hospitals (source), with ages;
  - a domain classifier tries to tell source images from images of the held-out hospital (target,
    images only). Its gradient reaches the features through a gradient reversal layer (GRL), which
    multiplies it by -lambda: the features are pushed to make the two domains indistinguishable while
    keeping the age prediction accurate on the source.

Choices (see docs/journal.md):
  - warm start from the source model (best-mae checkpoint): DANN fine-tunes it; the run it starts
    from gives the split, the age normalization and the "before" result;
  - age loss on normalized ages (MSE in z units, ~0.1-0.5) so that it has the same scale as the
    domain loss (binary cross-entropy, ~0.69): lambda then has its usual meaning;
  - BatchNorm statistics frozen (source statistics, as in the warm-start model): the alignment has
    to be learned by the weights, training and test use the same normalization, and batches of 8 + 8
    images do not make the statistics noisy. (AdaBN is the method that changes them.)
  - lambda schedule and learning-rate annealing of the original paper:
    lambda_p = lambda_max * (2 / (1 + exp(-10 p)) - 1),  lr_p = lr / (1 + 10 p) ** 0.75,
    with p going from 0 to 1 during training;
  - source images augmented as in the source run, target images never augmented;
  - transductive setting, as for AdaBN: all target images are used (without ages), then predicted;
  - --iw (IW-DANN, Tachet des Combes et al., NeurIPS 2020): against label shift, source images are
    re-weighted in the domain loss by w(age bin) = p_target(bin) / p_source(bin). p_target is
    estimated each epoch from the model's *predicted* ages on the target (true target ages are never
    used). Without it, DANN would also try to erase a real difference in age distribution
    (IOP is 7 years younger), which is what made AdaBN fail on IOP.

Model selection: without target ages, the main result is the final model (fixed training length,
as in the paper). The best source-validation checkpoint is also saved, as a check that DANN does
not degrade age prediction on the source.

Outputs: outputs/<source run>_dann<lambda>[_iw]/ with the same layout as a ClinicaDL run
(maps/.../final/model.pt, maps/.../best_models/best-mae/model.pt, split/, out_of_site.tsv), so
that predict_ages.py and adabn.py work with --run-dir, plus dann_log.tsv and dann_curves.png.

Run:  python scripts/train_dann.py --data t1linear --test-site IOP --augment --scheduler plateau \
          --lambda-max 0.3 [--iw]
      (the run options identify the source run, exactly as for predict_ages.py)
"""

import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_ages import IDS, load_model, model_path
from train_ixi_site_shift import AgeRegressor, add_run_args, age_table_path, make_dataset, run_dir_path

N_AGE_BINS = 5          # IW-DANN: source age quintiles
WEIGHT_RANGE = (0.25, 4.0)


class GradReverse(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by -lambda in the backward pass."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class DANN(nn.Module):
    def __init__(self, regressor: AgeRegressor):
        super().__init__()
        self.regressor = regressor  # DenseNet + age mean / std, saved alone at the end
        self.domain = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        head = self.regressor.net.class_layers
        return head.flatten(head.pool(head.relu(self.regressor.net.features(x))))

    def forward(self, x: torch.Tensor, lam: float = 0.0):
        """Normalized age z (age = mean + std * z) and domain logit (> 0: target)."""
        f = self.features(x)
        z = self.regressor.net.class_layers.out(f).squeeze(1)
        domain = self.domain(GradReverse.apply(f, lam)).squeeze(1)
        return z, domain


class Images(Dataset):
    """(image, age) pairs from a ClinicaDL dataset; age is 0 for target images (never used)."""

    def __init__(self, data: pd.DataFrame, images: str, augment: bool):
        self.ds = make_dataset(data.reset_index(drop=True), augment=augment, data=images)
        self.ds.train() if augment else self.ds.eval()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        sample = self.ds[i]
        img = sample.image.data if hasattr(sample.image, "data") else sample.image
        return img.float(), torch.tensor(float(sample.AgeMRI))


def freeze_batchnorm_stats(model: nn.Module) -> None:
    """Train mode for everything (dropout of the domain classifier) except BatchNorm statistics."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


@torch.no_grad()
def predict_ages(model: DANN, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    reg = model.regressor
    out = [(model(x.to(device))[0] * reg.age_std + reg.age_mean).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def iw_weights(pred_target: np.ndarray, edges: np.ndarray, p_source: np.ndarray) -> np.ndarray:
    """w(bin) = p_target / p_source, p_target from predicted target ages (+1 smoothing), clipped."""
    counts = np.histogram(np.clip(pred_target, edges[0], edges[-1]), bins=edges)[0] + 1.0
    w = counts / counts.sum() / p_source  # source-weighted mean of 1 before clipping
    return np.clip(w, *WEIGHT_RANGE)


def main():
    parser = argparse.ArgumentParser()
    add_run_args(parser)  # identifies the source run to start from
    parser.add_argument("--lambda-max", type=float, default=0.3, help="final weight of the reversed domain gradient")
    parser.add_argument("--iw", action="store_true", help="IW-DANN: re-weight source images by age (label shift)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8, help="per domain: each step sees 8 source + 8 target")
    parser.add_argument("--lr", type=float, default=3e-5, help="DenseNet and age head (fine-tuning)")
    parser.add_argument("--lr-domain", type=float, default=1e-4, help="domain classifier (trained from scratch)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    workers = args.workers if args.workers is not None else (2 if amp else 0)

    source_dir = run_dir_path(args)
    out_dir = source_dir.with_name(f"{source_dir.name}_dann{args.lambda_max:g}{'_iw' if args.iw else ''}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copytree(source_dir / "split", out_dir / "split")
    for name in ["out_of_site.tsv", "train_sites.tsv"]:
        shutil.copy(source_dir / name, out_dir / name)
    print(f"Source run : {source_dir.name}\nDANN run   : {out_dir.name}  ({device}, seed {args.seed})")

    # data: same train / validation fold as the source run; target = held-out hospital, without ages
    table = pd.read_csv(age_table_path(args.data), sep="\t")
    fold = source_dir / "split" / "5_fold" / "split-0"
    src_train = table.merge(pd.read_csv(fold / "train_baseline.tsv", sep="\t")[IDS], on=IDS)
    src_val = table.merge(pd.read_csv(fold / "validation_baseline.tsv", sep="\t")[IDS], on=IDS)
    target = pd.read_csv(source_dir / "out_of_site.tsv", sep="\t")[IDS].assign(AgeMRI=0.0)
    print(f"Source train {len(src_train)} | source val {len(src_val)} | target {len(target)} (ages not used)")

    loader = dict(batch_size=args.batch_size, num_workers=workers, pin_memory=amp)
    train_src = DataLoader(Images(src_train, args.data, args.augment), shuffle=True, drop_last=True, **loader)
    train_tgt = DataLoader(Images(target, args.data, False), shuffle=True, drop_last=True, **loader)
    val_src = DataLoader(Images(src_val, args.data, False), **loader)
    eval_tgt = DataLoader(Images(target, args.data, False), **loader)

    # model: warm start from the source model
    regressor = load_model(model_path(source_dir, "best-mae"), device)
    model = DANN(regressor).to(device)
    mean, std = float(regressor.age_mean), float(regressor.age_std)
    optimizer = torch.optim.Adam([
        {"params": model.regressor.parameters(), "lr": args.lr},
        {"params": model.domain.parameters(), "lr": args.lr_domain},
    ])
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # IW-DANN: age bins = source quintiles; weights updated from the predicted target ages
    edges = np.quantile(src_train.AgeMRI, np.linspace(0, 1, N_AGE_BINS + 1))
    edges[0], edges[-1] = 0.0, 120.0
    p_source = np.histogram(src_train.AgeMRI, bins=edges)[0] / len(src_train)
    weights = np.ones(N_AGE_BINS)

    total_steps = args.epochs * len(train_src)
    step, best_val, log = 0, math.inf, []
    pred_tgt = predict_ages(model, eval_tgt, device)
    val0 = np.abs(predict_ages(model, val_src, device) - src_val.AgeMRI.values).mean()
    print(f"Start (source model): source val MAE {val0:.2f} | mean predicted target age {pred_tgt.mean():.1f}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.iw:
            weights = iw_weights(pred_tgt, edges, p_source)
        w_bins = torch.tensor(weights, dtype=torch.float32, device=device)
        edges_t = torch.tensor(edges[1:-1], dtype=torch.float32, device=device)
        freeze_batchnorm_stats(model)
        tgt_iter = iter(train_tgt)
        sums = {"age": 0.0, "domain": 0.0, "acc": 0.0}
        for xs, ys in train_src:
            try:
                xt, _ = next(tgt_iter)
            except StopIteration:  # the target set is cycled
                tgt_iter = iter(train_tgt)
                xt, _ = next(tgt_iter)
            p = step / total_steps
            lam = args.lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr / (1.0 + 10.0 * p) ** 0.75

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            ns = len(xs)
            with torch.autocast(device_type=device.type, enabled=amp):
                z, dom = model(torch.cat([xs, xt]), lam)
            loss_age = F.mse_loss(z[:ns].float(), (ys - mean) / std)
            labels = torch.cat([torch.zeros(ns), torch.ones(len(xt))]).to(device)
            bce = F.binary_cross_entropy_with_logits(dom.float(), labels, reduction="none")
            w = w_bins[torch.bucketize(ys, edges_t, right=True)]  # same [a, b) bins as np.histogram
            loss_domain = 0.5 * (w * bce[:ns]).sum() / w.sum() + 0.5 * bce[ns:].mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss_age + loss_domain).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            sums["age"] += loss_age.item()
            sums["domain"] += loss_domain.item()
            sums["acc"] += ((dom.detach() > 0).float() == labels).float().mean().item()

        n = len(train_src)
        val_mae = np.abs(predict_ages(model, val_src, device) - src_val.AgeMRI.values).mean()
        pred_tgt = predict_ages(model, eval_tgt, device)
        row = {"epoch": epoch, "lambda": lam, "lr": optimizer.param_groups[0]["lr"],
               "loss_age_z": sums["age"] / n, "loss_domain": sums["domain"] / n,
               "domain_acc": sums["acc"] / n, "val_mae_source": val_mae,
               "target_pred_mean": pred_tgt.mean(), "target_pred_std": pred_tgt.std()}
        if args.iw:
            row |= {f"w{i}": weights[i] for i in range(N_AGE_BINS)}
        log.append(row)
        print(f"epoch {epoch:2d} | lambda {lam:.3f} | age loss {row['loss_age_z']:.3f} | domain loss "
              f"{row['loss_domain']:.3f} acc {row['domain_acc']:.2f} | source val MAE {val_mae:.2f} | "
              f"target mean pred {pred_tgt.mean():.1f} | {time.time() - t0:.0f} s")

        if val_mae < best_val:
            best_val = val_mae
            save(model.regressor, model_path(out_dir, "best-mae"))
    save(model.regressor, model_path(out_dir, "final"))

    log = pd.DataFrame(log)
    log.to_csv(out_dir / "dann_log.tsv", sep="\t", index=False, float_format="%.5g")
    plot_curves(log, val0, out_dir / "dann_curves.png")
    print(f"\nSource val MAE: start {val0:.2f}, final {log.val_mae_source.iloc[-1]:.2f}, best {best_val:.2f}")
    print(f"Results in {out_dir}  (evaluate with predict_ages.py --run-dir ... --checkpoint final)")


def save(regressor: AgeRegressor, path: Path) -> None:
    """Same format as ClinicaDL checkpoints ('network.' prefix), readable by predict_ages.load_model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({f"network.{k}": v.cpu() for k, v in regressor.state_dict().items()}, path)


def plot_curves(log: pd.DataFrame, val0: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(log.epoch, log.val_mae_source, marker="o")
    axes[0].axhline(val0, color="k", ls="--", lw=1, label="source model")
    axes[0].set(title="Source validation MAE", xlabel="epoch", ylabel="years")
    axes[0].legend()
    axes[1].plot(log.epoch, log.domain_acc, marker="o", label="domain accuracy")
    axes[1].axhline(0.5, color="k", ls="--", lw=1)
    axes[1].plot(log.epoch, log["lambda"], label="lambda")
    axes[1].set(title="Domain classifier (0.5 = domains indistinguishable)", xlabel="epoch")
    axes[1].legend()
    axes[2].plot(log.epoch, log.target_pred_mean, marker="o")
    axes[2].set(title="Mean predicted age on the target (no labels)", xlabel="epoch", ylabel="years")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
