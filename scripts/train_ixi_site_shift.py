"""
Âge cérébral sur IXI avec ClinicaDL 2.0 : effet du changement d'hôpital (site shift).

Expérience :
  - entraînement sur les hôpitaux TRAIN_SITES (split par patient : train / val / test interne)
  - évaluation 1 : test interne  (mêmes hôpitaux, sujets jamais vus)      -> "in-site"
  - évaluation 2 : tous les sujets de TEST_SITE (hôpital jamais vu)        -> "out-of-site"

Lancer :  python scripts/train_ixi_site_shift.py --test-site IOP
          python scripts/train_ixi_site_shift.py --check   (vérifie le prétraitement sur 1 image)
          python scripts/train_ixi_site_shift.py --test-site IOP --data t1linear   (Clinica t1-linear images)

--data bids2mm  (default) needs scripts/cache_preprocessed.py output in data/IXI/bids_2mm.
--data t1linear needs data/IXI/caps_t1linear_2mm (scripts/t1_linear_wsl.sh); results go to *_t1linear folders.
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
DATA_DIR = BASE_DIR / "data" / "IXI"
SITES = ["Guys", "HH", "IOP"]

# Two preprocessings of the same 563 images: (folder, files to read, transforms applied at load time).
# Both folders contain participants_sessions_age.tsv.
INPUTS = {
    # ToCanonical, Resample 2 mm, ZNormalization, CropOrPad 112x128x112, already applied once and for all
    # by scripts/cache_preprocessed.py
    "bids2mm": (DATA_DIR / "bids_2mm", BidsFileType(data_type="anat", suffix="T1w"), []),
    # Clinica t1-linear (N4 bias correction, affine registration to MNI, crop), then 2 mm (85x105x90) by
    # scripts/t1linear_to_2mm.py: padded to a size divisible by 16 (zeros = background), then z-normalized
    "t1linear": (
        DATA_DIR / "caps_t1linear_2mm",
        BidsFileType(data_type="t1_linear", suffix="T1w",
                     with_entities={"space": "MNI152NLin2009cSym", "desc": "Crop", "res": "2x2x2"}),
        [tio.CropOrPad((96, 112, 96)), tio.ZNormalization()],
    ),
}


def age_table_path(data: str = "bids2mm") -> Path:
    return INPUTS[data][0] / "participants_sessions_age.tsv"


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


def make_dataset(tsv: Path, augment: bool = False, data: str = "bids2mm") -> BidsDataset:
    folder, file_type, image_transforms = INPUTS[data]
    return BidsDataset(
        bids=folder,
        file_type=file_type,
        data=tsv,
        columns=["AgeMRI"],
        transforms=TransformsHandler(
            image_transforms=image_transforms,
            augmentations=AUGMENTATIONS if augment else [],
        ),
    )


def check_preprocessing(out_dir: Path, data: str = "bids2mm") -> None:
    """Sauve 3 coupes centrales d'un sujet par hôpital pour vérifier le cadrage."""
    import matplotlib.pyplot as plt

    table = pd.read_csv(age_table_path(data), sep="\t")
    for site in SITES:
        first = table[table.site == site].head(1)          # 1er sujet de cet hôpital
        tsv = out_dir / f"check_{site}.tsv"
        first.to_csv(tsv, sep="\t", index=False)
        dataset = make_dataset(tsv, data=data)
        dataset.eval()
        sample = dataset[0]

        img_obj = sample.image if hasattr(sample, "image") else sample["image"]
        tensor = img_obj.data if hasattr(img_obj, "data") else img_obj
        img = tensor.squeeze().numpy()
        print(f"{site} - {sample.participant_id} : forme {img.shape}, "
              f"moyenne {img.mean():.2f}, écart-type {img.std():.2f}")

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


def run_dir_path(test_site: str, augment: bool = False, data: str = "bids2mm") -> Path:
    """outputs/site_shift_test-<site>[_t1linear][_aug]: the bids2mm folders keep their original names."""
    suffix = ("" if data == "bids2mm" else f"_{data}") + ("_aug" if augment else "")
    return BASE_DIR / "outputs" / f"site_shift_test-{test_site}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-site", default="IOP", choices=SITES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--augment", action="store_true",
                        help="random scanner-contrast augmentation during training (results in *_aug folder)")
    parser.add_argument("--data", default="bids2mm", choices=list(INPUTS),
                        help="input images: bids2mm (own 2 mm preprocessing) or t1linear (Clinica t1-linear)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_dir = run_dir_path(args.test_site, args.augment, args.data)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_preprocessing(out_dir, args.data)
        return

    gpu = torch.cuda.is_available()
    computational = ComputationalConfig(gpu=gpu, amp=gpu, seed=42)
    print(f"GPU : {gpu}")

    # 1. Séparer hôpitaux d'entraînement / hôpital de test
    table = pd.read_csv(age_table_path(args.data), sep="\t")
    train_sites = [s for s in SITES if s != args.test_site]
    train_tsv = out_dir / "train_sites.tsv"
    ood_tsv = out_dir / "out_of_site.tsv"
    table[table.site.isin(train_sites)].to_csv(train_tsv, sep="\t", index=False)
    table[table.site == args.test_site].to_csv(ood_tsv, sep="\t", index=False)
    print(f"Entraînement sur {train_sites} : {table.site.isin(train_sites).sum()} sujets")
    print(f"Hôpital jamais vu {args.test_site} : {(table.site == args.test_site).sum()} sujets")

    # 2. Split par patient dans les hôpitaux d'entraînement
    dataset = make_dataset(train_tsv, augment=args.augment, data=args.data)
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
    print(f"Images : {args.data} ({INPUTS[args.data][0].name})")
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
    out_of_site = make_dataset(ood_tsv, data=args.data)
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
