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
- Much stronger than the IOP shift and affecting young subjects too.
- The plot of this run clipped a few HH predictions below 15 years (fixed since in `predict_ages.py`).

## Run v2-Guys (2026-10-28): Guys held out

- Same configuration as v2, but train on HH + IOP (249 subjects, 49 held out as internal test) and test on Guys (314 subjects).
- Stopped at epoch 33. Validation curve very noisy (MAE ~7-23 years); small training and validation sets.
- Drive folder: `outputs/site_shift_test-Guys`.

| | MAE | Mean signed error | Pearson r |
|---|---|---|---|
| insite (HH + IOP) | **7.51 years** | +5.27 | 0.91 |
| outofsite (Guys) | **17.13 years** | +16.93 | 0.81 |

Signed error by age group:

| Age | insite: n / MAE / bias | Guys: n / MAE / bias |
|---|---|---|
| ≤ 35 | 20 / 10.27 / +9.87 | 68 / 29.48 / +29.48 |
| 35-50 | 9 / 6.57 / +3.02 | 76 / 20.11 / +19.93 |
| 50-65 | 10 / 5.96 / +3.37 | 103 / 12.90 / +12.86 |
| 65+ | 10 / 4.36 / −0.01 | 67 / 7.71 / +7.05 |

- **Guys subjects are over-estimated**, strongly for young subjects and less with age: the opposite direction from HH.

## Summary of the held-out-hospital runs (v2 setup)

| Held out | Train subjects | insite MAE | outofsite MAE | outofsite bias | outofsite r |
|---|---|---|---|---|---|
| IOP | 495 | 5.81 | 10.83 | −5.74 | 0.67 |
| HH | 382 | 7.93 | 25.22 | −25.17 | 0.80 |
| Guys | 249 | 7.51 | 17.13 | +16.93 | 0.81 |

## Image statistics per site (2026-10-28, no training)

Computed locally on all 563 images: raw geometry, then intensity statistics inside the head (Otsu foreground, no skull stripping) after the current preprocessing; 3-class Otsu split of the head into dark / intermediate / bright tissue.

| Median | Guys | HH | IOP |
|---|---|---|---|
| Matrix / voxel (mm) | 256×256×150 / 0.94×0.94×1.2 | 256×256×150 / 0.94×0.94×1.2 | 256×256×146 / 0.94×0.94×1.2 |
| Head fraction of the volume (after preprocessing) | 0.234 | 0.226 | 0.181 |
| Head volume kept (ml) | 3004 | 2897 | 2329 |
| Mean / std inside head (z-units) | 1.39 / 0.59 | 1.46 / 0.58 | 1.57 / 0.83 |
| Bright − intermediate | 0.94 | 0.78 | 1.81 |
| Intermediate − dark | 0.70 | 0.64 | 0.86 |
| Fraction of bright voxels | 0.13 | 0.21 | 0.08 |

- **Same acquisition grid at all sites**, and Guys and HH have nearly identical head fraction and normalized mean/std: whole-volume normalization over background is **not** a plausible explanation for the Guys/HH shift.
- **Each site has a distinct contrast fingerprint**: Guys and HH barely overlap on bright − intermediate contrast; IOP differs strongly from both (heavy bright tail, higher spread, smaller detected head volume).
- These global measures are **nearly flat with age** within each site, and IOP (most different intensities) has the *smallest* shift: global intensity statistics do not explain the direction or size of the age bias. The cause is likely more local (how tissue contrast / atrophy appear on each scanner).
- Caveat: without skull stripping, the "bright" class mixes white matter and scalp fat.

## Clinica t1-linear preprocessing (2026-10-29)

- Clinica 0.12.0 `t1-linear` on all 563 images, locally (WSL Ubuntu, CPU, ANTsPy backend, `--random_seed 42`): N4 bias field correction, affine registration to MNI152NLin2009cSym, crop to 169×208×179 at 1 mm. Script: `scripts/t1_linear_wsl.sh`, resumable batches of 18 subjects with a checkpoint each (~17 min per batch, ~3.4 min per subject at 6 images × 2 threads; ~8 h in total).
- QC on 3 subjects (one per site): overlap of bright voxels with the MNI template (Dice) 0.87-0.91; IOP stays darker after t1-linear.
- Pitfall: ANTsPy leaves ~180 MB of registration temp files per subject in `/tmp` and never deletes them. The first run filled the disk at 324/563 (58 GB). The script now gives each batch its own temp folder, emptied after the batch, and stops before a batch if less than 10 GB is free.
- `scripts/t1linear_to_2mm.py` resamples the cropped images to 2 mm float32 (85×105×90, linear interpolation): 1.5 GB in total, uploaded to Drive as one `.tar`. At load time (`--data t1linear`): zero-padding to 96×112×96, then z-normalization.

## Run t1-linear (2026-10-29): same experiments on t1-linear images

- Same setup as v2 (DenseNet121, age normalized in the model, lr 1e-4, MSE loss, early stopping on validation MAE, `best-mae` checkpoint), only the input images change. Drive folders: `outputs/site_shift_test-<site>_t1linear`.
- IOP run: stopped at epoch 20 (best epoch 9). The validation curve is again very noisy (MAE 6-21 years from one epoch to the next), so the best-epoch choice is partly luck.

| Held out | insite MAE | outofsite MAE | outofsite bias | outofsite r | v2 outofsite MAE |
|---|---|---|---|---|---|
| IOP | 6.84 | 11.21 | −4.05 | 0.52 | 10.83 |
| HH | 7.12 | **8.62** | +4.82 | 0.85 | 25.22 |
| Guys | 7.04 | **8.87** | +2.51 | 0.75 | 17.13 |

Out-of-site signed error by age group (bias in years):

| Age | IOP: n / bias | HH: n / bias | Guys: n / bias |
|---|---|---|---|
| ≤ 35 | 31 / +6.20 | 56 / +14.94 | 68 / +14.19 |
| 35-50 | 20 / −3.20 | 38 / +6.23 | 76 / +6.53 |
| 50-65 | 6 / −15.91 | 57 / −0.75 | 103 / −1.21 |
| 65+ | 11 / −28.04 | 30 / −5.30 | 67 / −8.19 |

- **t1-linear removes most of the HH and Guys shift**: out-of-site MAE 25.2 → 8.6 (HH) and 17.1 → 8.9 (Guys), close to the in-site MAE (≈ 7). These drops are far larger than the run-to-run noise.
- The remaining HH / Guys error has the same shape as the in-site error (young subjects over-estimated, older ones under-estimated): regression to the mean, no longer a site-specific offset.
- **IOP is not fixed**: older IOP subjects are still strongly under-estimated (−28 years after 65) and the correlation drops (0.52). IOP has the most different contrast fingerprint (bright − intermediate 1.81 vs 0.78-0.94), which registration and N4 do not change.
- In-site MAE is similar to v2 (6.8-7.1 vs 5.8-7.9).
- Caveat: t1-linear combines N4 and affine registration; which of the two removes the Guys / HH shift is not known (an ablation could separate them). Single seed per run.

## Run v3 (to launch): scanner-contrast augmentation

- Same as v2, plus `--augment`: during training only, `RandomScannerContrast` rescales each image to [0, 1], applies a random gamma (log-uniform, 0.70-1.42) and, half of the time, a smooth bias field (coefficients 0.3), then maps the image back to its original intensity range. Applied with probability 0.8. Validation and test images are never augmented (ClinicaDL eval mode, checked).
- Why not re-normalize after the gamma: the cached images were cropped / padded after z-normalization, so re-normalizing shifts intensities (max difference 1.1 z-units even at gamma = 1) and training images would no longer match validation / test images. Mapping back to the original range keeps gamma = 1 an exact identity (difference 2e-7).
- Why this gamma range (calibrated on 12 Guys + 12 HH subjects, bright − intermediate contrast): gamma 0.7 → 1.4 moves the Guys median from 0.81 to 1.48 and HH from 0.68 to 1.18, i.e. about twice the Guys / HH gap (0.94 vs 0.78). IOP (1.81) stays out of reach of a reasonable gamma, so the augmentation may help less for IOP. Caveat: the range is close to TorchIO's default (log-gamma ±0.3), but its magnitude was sanity-checked on Guys and HH images, which are the held-out hospitals in two of the three runs. A strictly clean protocol would calibrate on the training hospitals of each run only.
- Results in `outputs/site_shift_test-<site>_aug`; compare with the v2 summary table above.
- After the t1-linear results, the more useful run is augmentation **on t1-linear images** (`--data t1linear --augment`, folders `..._t1linear_aug`): the geometric part of the shift is already removed, and IOP's remaining problem is contrast, which is what this augmentation targets.

## Interpretation so far

- **The model learns on its training hospitals**: in-site MAE 5.8-7.9 years vs ≈ 14 for the naive baseline, with the usual mild regression to the mean.
- **On an unseen hospital the error is large and systematic, and its direction depends on the hospital**: HH is under-estimated (−25 years on average), Guys over-estimated (+17), IOP under-estimated only for older subjects. "An unseen hospital makes brains look younger" is therefore wrong.
- **The ranking of subjects survives the shift** (r ≈ 0.7-0.8 on every unseen hospital): what breaks is a site-specific offset and slope (predicted ≈ a + b × true age, with a and b depending on the hospital).
- Hypothesis: some scanner-dependent image property is partly read by the model as age, with HH and Guys on opposite sides of it. The image statistics rule out field of view / background normalization for Guys vs HH and show clear per-site contrast fingerprints, but no global intensity measure explains the direction of the bias.
- The hypothesis "IOP is younger, so the gap is a population effect" is contradicted by the IOP run: the IOP gap comes from its older subjects.
- v2 improves both IOP MAEs over v1 and stabilizes training, but there is a single run each; differences of a few years may be chance.
- A train/validation gap appears in v2 (≈ 2.5 vs ≈ 6-7 years RMSE / MAE), i.e. some overfitting.
- **Update after t1-linear**: the Guys / HH shift was mostly geometric or bias-field related (removed by N4 + affine registration), not a global intensity effect. Two kinds of shift remain distinct: preprocessing fixes HH and Guys, while IOP's shift (contrast) survives it.

**Limitations**: one run per configuration; training sets of different sizes (249-495 subjects); small age groups in some test sets; no skull stripping; very noisy validation curves (the best-epoch choice is partly luck).

## Next ideas

After t1-linear, in order:

- **Stabilize training** before comparing methods: learning-rate scheduler, possibly larger batches or GroupNorm instead of BatchNorm, and several seeds per configuration.
- **Augmentation on t1-linear images** (`--data t1linear --augment`), aimed at IOP's contrast shift.
- **AdaBN** (recompute BatchNorm statistics on the unlabeled target images), then **DANN**.
- **Ablation N4 vs registration**, to know which step removes the Guys / HH shift.

Earlier ideas (before t1-linear):

- **Signed predictions for v1** (notebook cell 7 bis) to check whether v1 shows the same pattern.
- **Swap roles**: `--test-site Guys` and `--test-site HH`, to see whether the effect is specific to IOP.
- **Several seeds**: seeds are fixed to 42 (split and training); add a `--seed` option to measure run-to-run variability.
- **Intensity augmentation during training** (random gamma / contrast / bias field): first candidate after the image statistics, to make the model insensitive to each scanner's contrast fingerprint without re-preprocessing.
- **Histogram standardization** (landmarks learned on the training sites) to align intensity distributions across sites. Foreground-masked z-normalization alone is unlikely to help (Guys and HH already have the same normalized mean/std).
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
