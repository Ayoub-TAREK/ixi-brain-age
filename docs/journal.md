# Experiment journal

Working notes: configurations, raw results, tentative interpretations and next ideas.
All results are preliminary (a single training run per configuration).

## Setup common to all runs

- Training hospitals: Guys + HH, 495 subjects. 99 held out as the internal test set (`insite`); the rest is split with 5-fold CV and only fold 0 is used (train / validation).
- Unseen hospital: IOP, 68 subjects (`outofsite`).
- Preprocessing (precomputed by `scripts/cache_preprocessed.py`): ToCanonical, 2 mm resampling, z-normalization, CropOrPad 112 × 128 × 112. **No registration, no skull stripping.**
- Model: 3D DenseNet121 (MONAI), MSE loss, Adam, batch size 8, mixed precision.
- Early stopping (patience 10) on validation MAE; evaluation with the `best-mae` checkpoint.
- Google Colab T4 GPU: ~30 s per epoch.

**Populations**

| | Guys | HH | IOP |
|---|---|---|---|
| Subjects | 314 | 181 | 68 |
| Mean / median age | 50.8 / 53.4 | 47.4 / 48.0 | 42.4 / 36.3 |
| F / M | 175 / 139 | 94 / 87 | 44 / 24 |

**Naive baseline** (always predict the mean age of Guys + HH, 49.5 years): MAE 14.2 years on Guys + HH (all subjects, not only the internal test set), 16.2 years on IOP.

## Run v1 (2026-10-28): reference

- Network output = raw age in years, Adam lr 1e-3.
- Stopped at epoch 38, best epoch 27.
- Drive folder: `outputs/site_shift_test-IOP_v1_lr1e-3`.

| | MAE |
|---|---|
| insite | **6.16 years** |
| outofsite (IOP) | **12.21 years** |

MAE by age group:

| Age | insite | outofsite (IOP) |
|---|---|---|
| 20-35 | 4.73 (n=28) | 4.85 (n=30) |
| 35-50 | 8.49 (n=28) | 12.41 (n=20) |
| 50-65 | 6.30 (n=27) | 21.97 (n=6) |
| 65-90 | 4.37 (n=16) | 27.19 (n=11) |

(One 20.0-year-old IOP subject is missing: left bin edge excluded by `pd.cut`.)

Validation curve: very unstable, MAE oscillating between ~7 and ~52 years from one epoch to the next.

## Run v2 (2026-10-28): age normalization + lr 1e-4

- The network predicts a normalized age z; `AgeRegressor` maps it back to years (mean + std × z, statistics computed on train + validation only). Labels, loss and MAE stay in years, so results are directly comparable with v1.
- Adam lr 1e-4.
- Stopped at epoch 31, best epoch 20. Final training loss ≈ 6.5 years² (≈ 2.5 years RMSE).
- Drive folder: `outputs/site_shift_test-IOP`.

| | MAE | Mean signed error | Pearson r |
|---|---|---|---|
| insite | **5.81 years** | +1.83 | 0.89 |
| outofsite (IOP) | **10.83 years** | −5.74 | 0.67 |

Signed error (predicted − true age) by age group:

| Age | insite: n / MAE / bias | IOP: n / MAE / bias |
|---|---|---|
| ≤ 35 | 28 / 6.57 / +5.32 | 31 / 5.57 / +4.41 |
| 35-50 | 28 / 7.16 / +2.82 | 20 / 6.99 / −5.15 |
| 50-65 | 27 / 4.33 / +0.28 | 6 / 18.36 / −18.36 |
| 65+ | 16 / 4.66 / −3.41 | 11 / 28.53 / −28.53 |

Validation curve: much more stable than v1 (MAE between ~6 and ~22 years), but still noisy from one epoch to the next.

## Run v2-HH (2026-10-28): HH held out

- Same configuration as v2, but train on Guys + IOP (382 subjects, 76 held out as internal test) and test on HH (181 subjects).
- Stopped at epoch 28. Validation curve noisy (MAE ~7-18 years, one spike at ~30).
- Drive folder: `outputs/site_shift_test-HH`.

| | MAE | Mean signed error | Pearson r |
|---|---|---|---|
| insite (Guys + IOP) | **7.93 years** | −2.43 | 0.78 |
| outofsite (HH) | **25.22 years** | −25.17 | 0.80 |

Signed error by age group:

| Age | insite: n / MAE / bias | HH: n / MAE / bias |
|---|---|---|
| ≤ 35 | 19 / 7.03 / +3.17 | 56 / 14.42 / −14.24 |
| 35-50 | 17 / 8.92 / −1.26 | 38 / 22.93 / −22.93 |
| 50-65 | 31 / 7.12 / −4.23 | 57 / 31.82 / −31.82 |
| 65+ | 9 / 10.74 / −10.23 | 30 / 35.76 / −35.76 |

- **All HH subjects are under-estimated**, increasingly with age. Predictions are compressed and shifted down (roughly predicted ≈ 15 + 0.4 × true age), yet the ranking is preserved (r = 0.80, as high as in-site).
- Same failure mode as IOP (unseen hospital → brains look "younger"), but much stronger and affecting young subjects too.
- The plot of this run clipped a few HH predictions below 15 years (fixed since in `predict_ages.py`).

## Interpretation so far

- **The model learns on its training hospitals**: MAE ≈ 6 years vs 14 for the naive baseline, well calibrated, with the usual mild regression to the mean (young subjects slightly over-estimated, old ones slightly under-estimated).
- **On the unseen hospital, the error is not uniform**: young IOP subjects are predicted as well as young Guys/HH subjects, but **older IOP subjects are systematically and strongly under-estimated** (every IOP subject above ~55 is predicted 20 to 45 years too young). Predicted IOP ages flatten out around 40-50 years.
- The hypothesis "IOP is younger, so the gap is a population effect" is contradicted: the gap comes from the older IOP subjects.
- Hypothesis (not tested): the ageing markers learned by the model (atrophy, gray/white matter contrast, ventricle size) look "younger" on IOP images, e.g. because of a different scanner contrast or intensity profile.
- v2 improves both MAEs and stabilizes training, but this is a single run each; the difference may be partly due to chance.
- New in v2: a train/validation gap (≈ 2.5 vs ≈ 6-7 years), i.e. overfitting is now visible.

**Limitations**: one run per configuration, small IOP age groups above 50 (6 and 11 subjects), one hospital pair tested, no registration or skull stripping.

## Next ideas

- **Signed predictions for v1** (notebook cell 7 bis) to check whether v1 shows the same pattern.
- **Swap roles**: `--test-site Guys` and `--test-site HH`, to see whether the effect is specific to IOP.
- **Several seeds**: seeds are fixed to 42 (split and training); add a `--seed` option to measure run-to-run variability.
- **Intensity normalization** (main suspect after the HH run): `ZNormalization` currently uses the whole volume, background and skull included, so differences in field of view, background or contrast between scanners shift the normalized brain intensities. Try foreground-masked z-normalization or histogram standardization, then rerun the IOP and HH tests.
- **Linear recalibration on the new site**: since the ranking is preserved (r ≈ 0.8), check how much of the error a linear fit on a few labeled subjects of the new hospital removes.
- **Look at the images**: compare intensity histograms and gray/white matter contrast of older subjects across hospitals, to test the scanner-contrast hypothesis.
- **Bias correction**, standard in brain age studies (linear regression of the error on true age, fitted on validation).
- **Robustness to scanners**: intensity / contrast augmentations (in `IMAGE_TRANSFORMS`), site harmonization.
- **Regularization** now that overfitting appears: augmentations, weight decay, dropout.
- **More complete preprocessing**: registration (e.g. Clinica `t1-linear`) and skull stripping.
- **Practical**: keep `bids_2mm` on Drive to skip the preprocessing at each session.

## Technical pitfalls (ClinicaDL 2.0)

- Test group names must not contain hyphens or underscores (`insite`, not `in-site`): a hyphen breaks MAPS parsing.
- `OptimizationConfig(num_epochs=...)`; `make_split` writes `train.tsv`; `KFold` takes the folder returned by `make_kfold`; `ModelCheckpointCallback` epochs start at 1.
- Early stopping must monitor a computed metric (`mae`, not `loss`); use `MAEMetricConfig(label_key="AgeMRI")`.
- `trainer.test()` assumes a GPU by default: pass it the same `ComputationalConfig`.
- On CPU, `amp` and `channels_last` make 3D convolutions extremely slow.
- ClinicaDL refuses to overwrite an existing MAPS or split folder (hence `clean_run()` in the notebook).
- The test step only saves the absolute error per subject (`details.tsv`), not predictions (hence `predict_ages.py`).
- Checkpoints: `maps/training/split-0/models/best_models/best-mae/model.pt`, keys prefixed with `network.`.
- `git pull` in Colab updates the scripts, not the open notebook: reopen the notebook from GitHub.
- Precomputing the preprocessing: image loading 0.53 s → 0.08 s (measured locally).
