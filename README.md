# ixi-brain-age

Brain age prediction from T1-weighted MRI of the [IXI](https://brain-development.org/ixi-dataset/) dataset, using [ClinicaDL 2.0](https://clinicadl.readthedocs.io/) and a 3D DenseNet (MONAI).

The goal is to measure, and then reduce, the effect of a **domain shift between hospitals** (site shift). IXI comes from 3 London hospitals (Guys, HH, IOP); each one is held out in turn:

- training on the two other hospitals (subject-level split: train / validation / internal test);
- evaluation on the internal test set (same hospitals, unseen subjects): **in-site**;
- evaluation on the held-out hospital, never seen during training: **out-of-site**.

## Preprocessing

Two preprocessings of the same images are compared:

- **Baseline**: resampling to 2 mm, z-normalization and crop, computed once by `scripts/cache_preprocessed.py` (`data/IXI/bids_2mm`). No registration.
- **Clinica t1-linear**: N4 bias field correction and affine registration to the MNI template, run locally by `scripts/t1_linear_wsl.sh` (~8 h on CPU for 563 images), then resampled to 2 mm by `scripts/t1linear_to_2mm.py` (`data/IXI/caps_t1linear_2mm`). Selected with `--data t1linear`.

One image per hospital after the baseline preprocessing (output of `--check`):

| Guys | HH | IOP |
|:---:|:---:|:---:|
| ![Guys](docs/check_Guys_sub-IXI002.png) | ![HH](docs/check_HH_sub-IXI012.png) | ![IOP](docs/check_IOP_sub-IXI035.png) |

## Structure

```
scripts/ixi_to_bids.py                IXI -> BIDS conversion + age table
scripts/cache_preprocessed.py         one-off baseline preprocessing (2 mm, normalization, crop)
scripts/t1_linear_wsl.sh              Clinica t1-linear on all subjects (Linux / WSL, resumable batches)
scripts/t1linear_to_2mm.py            2 mm copy of the t1-linear images
scripts/train_ixi_site_shift.py       training and in-site / out-of-site evaluation
scripts/predict_ages.py               signed predictions (predicted - true age) and plots
notebooks/ixi_brain_age_colab.ipynb   end-to-end run on Google Colab
docs/                                 preprocessing check images, experiment journal
```

## Reproduce

The IXI data is **not included** in this repository. Download it from the official source: https://brain-development.org/ixi-dataset/ (see the terms of use on that site).

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download `IXI-T1.tar` and `IXI.xls`, and place them as follows:
   ```
   data/IXI/raw/IXI002-Guys-0828-T1.nii.gz ...   (extracted from IXI-T1.tar)
   data/IXI/IXI.xls
   ```
3. Convert to BIDS (output in `data/IXI/bids/`):
   ```bash
   python scripts/ixi_to_bids.py
   ```
4. Precompute the preprocessing (output in `data/IXI/bids_2mm/`):
   ```bash
   python scripts/cache_preprocessed.py
   ```
5. Check the preprocessing on one image per hospital:
   ```bash
   python scripts/train_ixi_site_shift.py --check
   ```
6. Train and evaluate (options: `--epochs`, `--batch-size`, `--lr`):
   ```bash
   python scripts/train_ixi_site_shift.py --test-site IOP
   ```
7. Compute signed predictions and plots:
   ```bash
   python scripts/predict_ages.py --test-site IOP
   ```

Results are written to `outputs/site_shift_test-IOP/`.

For the t1-linear images: run Clinica t1-linear with `bash scripts/t1_linear_wsl.sh` (Linux / WSL, Clinica installed in `~/clinica_env`), then add `--data t1linear` to steps 5-7 (results in `outputs/site_shift_test-<site>_t1linear/`).

## Results

Each hospital is held out in turn. Mean absolute error (MAE) in years:

| Held-out hospital | In-site (baseline) | Out-of-site (baseline) | In-site (t1-linear) | Out-of-site (t1-linear) |
|---|---|---|---|---|
| IOP | 5.8 | 10.8 | 6.8 | 11.2 |
| HH | 7.9 | **25.2** | 7.1 | **8.6** |
| Guys | 7.5 | **17.1** | 7.0 | **8.9** |

For reference, always predicting the mean training age gives an MAE of about 14-16 years.

- **Baseline**: the error is 2 to 3 times higher on an unseen hospital, and systematic: HH subjects are all predicted too young, Guys subjects too old, older IOP subjects too young. Subjects are still ranked correctly by age (correlation ≈ 0.7-0.8): the model has partly learned each scanner's image style.
- **t1-linear** (bias field correction + registration) **removes most of the HH and Guys shift**: their out-of-site error becomes close to the in-site error.
- **IOP is not fixed**: older IOP subjects are still predicted much too young. IOP has a very different tissue contrast, which registration does not change.

Next steps: stabilize training (noisy validation curves, several seeds), contrast augmentation on t1-linear images (aimed at IOP), then unsupervised domain adaptation (AdaBN, DANN). Single run per configuration; details, plots and analyses in [docs/journal.md](docs/journal.md).
