# ixi-brain-age

Brain age prediction from T1-weighted MRI of the [IXI](https://brain-development.org/ixi-dataset/) dataset, using [ClinicaDL 2.0](https://clinicadl.readthedocs.io/) and a 3D DenseNet (MONAI).

The goal is to measure the effect of a **domain shift between hospitals** (site shift):

- training on the **Guys** and **HH** hospitals (subject-level split: train / validation / internal test);
- evaluation on the internal test set (same hospitals, unseen subjects);
- evaluation on the **IOP** hospital, never seen during training.

## Preprocessing check

Preprocessing is computed once by `scripts/cache_preprocessed.py` (2 mm images in `data/IXI/bids_2mm`), which speeds up training.

One image per hospital after preprocessing (output of `--check`):

| Guys | HH | IOP |
|:---:|:---:|:---:|
| ![Guys](docs/check_Guys_sub-IXI002.png) | ![HH](docs/check_HH_sub-IXI012.png) | ![IOP](docs/check_IOP_sub-IXI035.png) |

## Structure

```
scripts/ixi_to_bids.py                IXI -> BIDS conversion + age table
scripts/cache_preprocessed.py         one-off preprocessing (2 mm, normalization, crop)
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

## Results

Baseline (current preprocessing, no registration): each hospital is held out in turn. The model is trained on the two others and tested on unseen subjects from the training hospitals (in-site) and on the held-out hospital (out-of-site). Mean absolute error (MAE) in years:

| Held-out hospital | In-site MAE | Out-of-site MAE | Out-of-site error |
|---|---|---|---|
| IOP | 5.8 | 10.8 | older subjects predicted too young |
| HH | 7.9 | 25.2 | all subjects predicted too young |
| Guys | 7.5 | 17.1 | all subjects predicted too old |

For reference, always predicting the mean training age gives an MAE of about 14-16 years.

- The model learns brain age on its training hospitals, but its error is **2 to 3 times higher on an unseen hospital**.
- The error is **systematic, and its direction depends on the hospital**, while subjects are still ranked correctly by age (correlation ≈ 0.7-0.8): the model has partly learned each scanner's image style.

Next steps: Clinica `t1-linear` preprocessing, then data augmentation and unsupervised domain adaptation (e.g. DANN) to reduce the gap. Single run per configuration; details, plots and analyses in [docs/journal.md](docs/journal.md).
