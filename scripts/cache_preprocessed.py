"""
Pré-calcule le prétraitement des IRM une seule fois (au lieu de le refaire à chaque epoch).

Entrée  : data/IXI/bids/       (sortie de scripts/ixi_to_bids.py)
Sortie  : data/IXI/bids_2mm/   (même structure BIDS, images prétraitées)
            sub-IXI002/ses-M000/anat/sub-IXI002_ses-M000_T1w.nii.gz   (112 x 128 x 112, 2 mm)
            participants_sessions_age.tsv, dataset_description.json   (copiés tels quels)

Les images déjà traitées sont sautées : le script peut être relancé sans risque.

Lancer depuis la racine du projet :  python scripts/cache_preprocessed.py
"""

import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import torchio as tio

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "data" / "IXI" / "bids"
DST_DIR = BASE_DIR / "data" / "IXI" / "bids_2mm"

# Même prétraitement, dans le même ordre, que l'ancien IMAGE_TRANSFORMS de train_ixi_site_shift.py
# (pas de recalage ni de retrait du crâne : limite à mentionner dans le README)
PREPROCESSING = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(2),                              # 2 mm isotrope
    tio.ZNormalization(),
    tio.CropOrPad(target_shape=(112, 128, 112)),
])


def is_image(path: Path) -> bool:
    return path.name.endswith(("_T1w.nii", "_T1w.nii.gz"))


def preprocess(src: Path, dst: Path) -> None:
    """Prétraite une image et l'écrit dans dst (via un fichier temporaire : pas de fichier à moitié écrit)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("tmp_" + dst.name)
    PREPROCESSING(tio.ScalarImage(src)).save(tmp)
    os.replace(tmp, dst)


def _job(paths: tuple[Path, Path]) -> None:
    preprocess(*paths)


def main():
    files = [
        p for p in SRC_DIR.rglob("*")
        if p.is_file() and "derivatives" not in p.relative_to(SRC_DIR).parts
    ]

    # fichiers non-image (tsv, json...) : recopiés à chaque fois pour rester à jour
    for src in files:
        if not is_image(src):
            dst = DST_DIR / src.relative_to(SRC_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    images = [p for p in files if is_image(p)]
    todo = [
        (src, DST_DIR / src.relative_to(SRC_DIR))
        for src in images
        if not (DST_DIR / src.relative_to(SRC_DIR)).exists()
    ]
    print(f"Images : {len(images)} au total, {len(images) - len(todo)} déjà traitées, {len(todo)} à traiter")

    with Pool(processes=os.cpu_count()) as pool:
        for i, _ in enumerate(pool.imap_unordered(_job, todo), start=1):
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")

    print(f"Terminé : {DST_DIR}")


if __name__ == "__main__":
    main()
