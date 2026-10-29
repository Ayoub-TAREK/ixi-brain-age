"""
Compact 2 mm copy of the Clinica t1-linear outputs, for training on Colab.

Input  : <caps>/subjects/sub-*/ses-*/t1_linear/*_desc-Crop_res-1x1x1_T1w.nii.gz   (169 x 208 x 179, 1 mm, float64)
Output : <caps_2mm>/subjects/sub-*/ses-*/t1_linear/*_desc-Crop_res-2x2x2_T1w.nii.gz (2 mm, float32)
         + dataset_description.json and participants_sessions_age.tsv

Only the resolution and storage type change (linear interpolation, like the current 2 mm pipeline):
intensity normalization and cropping are left to the training code. Already converted images are
skipped. A 1 mm image that cannot be read is deleted, so that the next t1-linear pass recomputes it.

Usage (in the Clinica environment):  python t1linear_to_2mm.py <caps> <caps_2mm> <age_tsv>
"""

import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def convert(src: Path, dst: Path) -> None:
    out = resample_to_output(nib.load(src), voxel_sizes=(2.0, 2.0, 2.0), order=1)
    img = nib.Nifti1Image(out.get_fdata(dtype=np.float32), out.affine)
    img.header.set_data_dtype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("tmp_" + dst.name)  # no half-written file if interrupted
    nib.save(img, tmp)
    tmp.replace(dst)


def main():
    caps, caps_2mm, age_tsv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    caps_2mm.mkdir(parents=True, exist_ok=True)
    if (caps / "dataset_description.json").exists():
        shutil.copy2(caps / "dataset_description.json", caps_2mm)
    shutil.copy2(age_tsv, caps_2mm)

    converted, broken = 0, 0
    for src in sorted(caps.glob("subjects/*/*/t1_linear/*_desc-Crop_res-1x1x1_T1w.nii.gz")):
        dst = caps_2mm / src.relative_to(caps).parent / src.name.replace("res-1x1x1", "res-2x2x2")
        if dst.exists():
            continue
        try:
            convert(src, dst)
            converted += 1
        except Exception as e:  # unreadable output: remove it so that t1-linear redoes this subject
            print(f"  cannot read {src.name} ({e}): deleted, will be recomputed")
            src.unlink(missing_ok=True)
            broken += 1
    total = len(list(caps_2mm.glob("subjects/*/*/t1_linear/*_res-2x2x2_T1w.nii.gz")))
    print(f"  2 mm copy: {converted} new, {broken} broken, {total} in total")


if __name__ == "__main__":
    main()
