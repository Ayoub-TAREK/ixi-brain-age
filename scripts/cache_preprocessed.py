"""
Computes the baseline (v1) MRI preprocessing once, instead of at every epoch.

Input   : data/IXI/bids/       (output of scripts/ixi_to_bids.py)
Output  : data/IXI/bids_2mm/   (same BIDS layout, preprocessed images)
            sub-IXI002/ses-M000/anat/sub-IXI002_ses-M000_T1w.nii.gz   (112 x 128 x 112, 2 mm)
            participants_sessions_age.tsv, dataset_description.json   (copied as is)

Images already processed are skipped: the script can safely be run again.

Run from the project root:  python scripts/cache_preprocessed.py
"""

import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import torchio as tio

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "data" / "IXI" / "bids"
DST_DIR = BASE_DIR / "data" / "IXI" / "bids_2mm"

# Same preprocessing, in the same order, as the former IMAGE_TRANSFORMS of train_ixi_site_shift.py
# (no registration, no skull stripping: see the README)
PREPROCESSING = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(2),                              # 2 mm isotropic
    tio.ZNormalization(),
    tio.CropOrPad(target_shape=(112, 128, 112)),
])


def is_image(path: Path) -> bool:
    return path.name.endswith(("_T1w.nii", "_T1w.nii.gz"))


def preprocess(src: Path, dst: Path) -> None:
    """Preprocesses one image and writes it to dst (through a temporary file: never a half-written file)."""
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

    # non-image files (tsv, json...): copied again every time to stay up to date
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
    print(f"Images: {len(images)} in total, {len(images) - len(todo)} already done, {len(todo)} to process")

    with Pool(processes=os.cpu_count()) as pool:
        for i, _ in enumerate(pool.imap_unordered(_job, todo), start=1):
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")

    print(f"Done: {DST_DIR}")


if __name__ == "__main__":
    main()
