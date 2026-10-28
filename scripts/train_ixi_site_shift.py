"""
Âge cérébral sur IXI avec ClinicaDL 2.0 : effet du changement d'hôpital (site shift).

Expérience :
  - entraînement sur les hôpitaux TRAIN_SITES (split par patient : train / val / test interne)
  - évaluation 1 : test interne  (mêmes hôpitaux, sujets jamais vus)      -> "in-site"
  - évaluation 2 : tous les sujets de TEST_SITE (hôpital jamais vu)        -> "out-of-site"

Lancer :  python scripts/train_ixi_site_shift.py --test-site IOP
          python scripts/train_ixi_site_shift.py --check   (vérifie le prétraitement sur 1 image)

Nécessite la sortie de scripts/cache_preprocessed.py dans data/IXI/bids_2mm.
Les corrections d'API trouvées sur brain_age_test.py sont reprises ici.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torchio as tio
from monai.networks.nets import DenseNet121

from clinicadl.callbacks import EarlyStoppingCallback, ModelCheckpointCallback
from clinicadl.data.dataloader import DataLoader
from clinicadl.data.datasets import BidsDataset
from clinicadl.io.bids import BidsFileType
from clinicadl.metrics.config import MAEMetricConfig
from clinicadl.models import SupervisedModel
from clinicadl.optim import OptimizationConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.split import KFold, make_kfold, make_split
from clinicadl.train import ComputationalConfig, Trainer
from clinicadl.transforms import TransformsHandler

BASE_DIR = Path(__file__).resolve().parent.parent
BIDS_DIR = BASE_DIR / "data" / "IXI" / "bids_2mm"
ALL_TSV = BIDS_DIR / "participants_sessions_age.tsv"
SITES = ["Guys", "HH", "IOP"]

# Le prétraitement (ToCanonical, Resample 2 mm, ZNormalization, CropOrPad 112x128x112) est déjà
# appliqué une fois pour toutes par scripts/cache_preprocessed.py.
# C'est ici qu'on ajoutera les futures augmentations de données.
IMAGE_TRANSFORMS = []


class RandomScannerContrast(tio.Transform):
    """Training-only augmentation that simulates another scanner's tissue contrast.

    The image is rescaled to [0, 1], a random gamma (contrast between tissues) and a smooth
    bias field are applied, then the image is mapped back to its original intensity range:
    with gamma = 1 and no bias field the image is unchanged, so augmented training images
    stay on the same scale as validation / test images. The default gamma range (0.70-1.42)
    is about twice the Guys / HH contrast difference measured in docs/journal.md.
    """

    def __init__(self, log_gamma: float = 0.35, bias_coefficients: float = 0.3, bias_p: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.log_gamma = log_gamma
        self.bias_field = tio.RandomBiasField(coefficients=bias_coefficients, p=bias_p)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in subject.get_images(intensity_only=True):
            x = image.data.float()
            lo, hi = x.min(), x.max()
            gamma = torch.empty(1).uniform_(-self.log_gamma, self.log_gamma).exp()
            u = tio.ScalarImage(tensor=((x - lo) / (hi - lo)) ** gamma, affine=image.affine)
            image.set_data(lo + (hi - lo) * self.bias_field(u).data)
        return subject


# Applied by ClinicaDL to training images only (validation and test images are never augmented)
AUGMENTATIONS = [RandomScannerContrast(p=0.8)]


class AgeRegressor(torch.nn.Module):
    """DenseNet qui prédit un âge normalisé z ; la sortie est remise en années (moyenne + écart-type * z).

    Les labels, la loss et la MAE restent donc en années. La moyenne et l'écart-type sont
    enregistrés dans le checkpoint (buffers), ce qui permet de recharger le modèle tel quel.
    """

    def __init__(self, age_mean: float = 0.0, age_std: float = 1.0):
        super().__init__()
        self.net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
        self.register_buffer("age_mean", torch.tensor(float(age_mean)))
        self.register_buffer("age_std", torch.tensor(float(age_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.age_std + self.age_mean


def make_dataset(tsv: Path, augment: bool = False) -> BidsDataset:
    return BidsDataset(
        bids=BIDS_DIR,
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
        data=tsv,
        columns=["AgeMRI"],
        transforms=TransformsHandler(
            image_transforms=IMAGE_TRANSFORMS,
            augmentations=AUGMENTATIONS if augment else [],
        ),
    )


def check_preprocessing(out_dir: Path) -> None:
    """Sauve 3 coupes centrales d'un sujet par hôpital pour vérifier le cadrage."""
    import matplotlib.pyplot as plt

    table = pd.read_csv(ALL_TSV, sep="\t")
    for site in SITES:
        first = table[table.site == site].head(1)          # 1er sujet de cet hôpital
        tsv = out_dir / f"check_{site}.tsv"
        first.to_csv(tsv, sep="\t", index=False)
        dataset = make_dataset(tsv)
        dataset.eval()
        sample = dataset[0]

        img_obj = sample.image if hasattr(sample, "image") else sample["image"]
        tensor = img_obj.data if hasattr(img_obj, "data") else img_obj
        img = tensor.squeeze().numpy()
        print(f"{site} - {sample.participant_id} : forme {img.shape}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        slices = [img[img.shape[0] // 2], img[:, img.shape[1] // 2], img[:, :, img.shape[2] // 2]]
        for ax, sl in zip(axes, slices):
            ax.imshow(sl.T, cmap="gray", origin="lower")
            ax.axis("off")
        fig.suptitle(f"{site} - {sample.participant_id}")
        path = out_dir / f"check_{site}_{sample.participant_id}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-site", default="IOP", choices=SITES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true",
                        help="random scanner-contrast augmentation during training (results in *_aug folder)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_dir = BASE_DIR / "outputs" / f"site_shift_test-{args.test_site}{'_aug' if args.augment else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_preprocessing(out_dir)
        return

    gpu = torch.cuda.is_available()
    computational = ComputationalConfig(gpu=gpu, amp=gpu, seed=42)
    print(f"GPU : {gpu}")

    # 1. Séparer hôpitaux d'entraînement / hôpital de test
    table = pd.read_csv(ALL_TSV, sep="\t")
    train_sites = [s for s in SITES if s != args.test_site]
    train_tsv = out_dir / "train_sites.tsv"
    ood_tsv = out_dir / "out_of_site.tsv"
    table[table.site.isin(train_sites)].to_csv(train_tsv, sep="\t", index=False)
    table[table.site == args.test_site].to_csv(ood_tsv, sep="\t", index=False)
    print(f"Entraînement sur {train_sites} : {table.site.isin(train_sites).sum()} sujets")
    print(f"Hôpital jamais vu {args.test_site} : {(table.site == args.test_site).sum()} sujets")

    # 2. Split par patient dans les hôpitaux d'entraînement
    dataset = make_dataset(train_tsv, augment=args.augment)
    split_dir = make_split(train_tsv, n_test=0.2, seed=42)
    kfold_dir = make_kfold(split_dir / "train.tsv", n_splits=5, seed=42)
    split = next(iter(KFold(kfold_dir).get_splits(dataset)))  # on n'utilise que le 1er fold
    split.build_train_loader(batch_size=args.batch_size, shuffle=True)
    split.build_val_loader(batch_size=args.batch_size)

    # 3. Modèle et entraînement
    # normalisation de l'âge avec les sujets train + validation uniquement (le test interne est exclu)
    pool_ids = pd.read_csv(split_dir / "train.tsv", sep="\t")[["participant_id", "session_id"]]
    train_ages = table.merge(pool_ids, on=["participant_id", "session_id"])["AgeMRI"]
    print(f"Âge (train + val) : moyenne {train_ages.mean():.1f}, écart-type {train_ages.std():.1f}")
    print(f"Learning rate : {args.lr}")
    print(f"Augmentation : {AUGMENTATIONS if args.augment else 'none'}")
    model = SupervisedModel(
        network=AgeRegressor(train_ages.mean(), train_ages.std()),
        loss=torch.nn.MSELoss(),
        optimizer=AdamConfig(lr=args.lr),
        label_key="AgeMRI",
    )
    trainer = Trainer(
        maps=out_dir / "maps",
        model=model,
        callbacks=[
            EarlyStoppingCallback(metric="mae", patience=10),
            # sauve aussi le meilleur modèle en validation (MAE), évalué plus bas
            ModelCheckpointCallback(metric="mae", epochs=range(10, args.epochs + 1, 10)),
        ],
        metrics={"mae": MAEMetricConfig(label_key="AgeMRI")},
        optimization=OptimizationConfig(num_epochs=args.epochs),
    )
    trainer.train(split=split, computational=computational)

    # 4. Évaluations : même hôpitaux vs hôpital jamais vu
    in_site = dataset.subset(split_dir / "test_baseline.tsv")
    out_of_site = make_dataset(ood_tsv)
    for name, ds in [("insite", in_site), ("outofsite", out_of_site)]:
        print(f"\n=== Test {name} ({len(ds)} images) ===")
        trainer.test(
            model_checkpoint="split-0_best-mae",  # meilleure MAE de validation, pas la dernière epoch
            group_name=name,
            dataloader=DataLoader(ds),
            metrics=["mae"],
            computational=computational,
        )
    print(f"\nRésultats dans {out_dir / 'maps'}")


if __name__ == "__main__":
    main()
