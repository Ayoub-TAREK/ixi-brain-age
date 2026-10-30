# Experiment journal

Working notes: configurations, raw results, tentative interpretations and next ideas.
All results are preliminary (a single training run per configuration).

Naming, following the project plan: **v1** = baseline (own preprocessing; v1a was a first try with lr 1e-3), **v2** = Clinica t1-linear preprocessing, **v3a** = contrast augmentation on t1-linear images, **v3b** = unsupervised domain adaptation (AdaBN, DANN; to do).

## Setup common to all runs

Described for the IOP held-out runs; the HH and Guys runs swap the roles of the hospitals.

- Training hospitals: Guys + HH, 495 subjects. 99 held out as the internal test set (`insite`); the rest is split with 5-fold CV and only fold 0 is used (train / validation).
- Unseen hospital: IOP, 68 subjects (`outofsite`).
- Preprocessing in v1 (precomputed by `scripts/cache_preprocessed.py`): ToCanonical, 2 mm resampling, z-normalization, CropOrPad 112 × 128 × 112. **No registration, no skull stripping.** (v2 and v3: Clinica t1-linear, see below.)
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

## v1a (2026-10-28): first baseline run (lr 1e-3)

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

## v1 (2026-10-28): baseline, age normalization + lr 1e-4

- The network predicts a normalized age z; `AgeRegressor` maps it back to years (mean + std × z, statistics computed on train + validation only). Labels, loss and MAE stay in years, so results are directly comparable with v1a.
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

Validation curve: much more stable than v1a (MAE between ~6 and ~22 years), but still noisy from one epoch to the next.

## v1-HH (2026-10-28): HH held out

- Same configuration as v1, but train on Guys + IOP (382 subjects, 76 held out as internal test) and test on HH (181 subjects).
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

## v1-Guys (2026-10-28): Guys held out

- Same configuration as v1, but train on HH + IOP (249 subjects, 49 held out as internal test) and test on Guys (314 subjects).
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

## Summary of the v1 baseline runs

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

## v2: Clinica t1-linear preprocessing (2026-10-29)

- Clinica 0.12.0 `t1-linear` on all 563 images, locally (WSL Ubuntu, CPU, ANTsPy backend, `--random_seed 42`): N4 bias field correction, affine registration to MNI152NLin2009cSym, crop to 169×208×179 at 1 mm. Script: `scripts/t1_linear_wsl.sh`, resumable batches of 18 subjects with a checkpoint each (~17 min per batch, ~3.4 min per subject at 6 images × 2 threads; ~8 h in total).
- QC on 3 subjects (one per site): overlap of bright voxels with the MNI template (Dice) 0.87-0.91; IOP stays darker after t1-linear.
- Pitfall: ANTsPy leaves ~180 MB of registration temp files per subject in `/tmp` and never deletes them. The first run filled the disk at 324/563 (58 GB). The script now gives each batch its own temp folder, emptied after the batch, and stops before a batch if less than 10 GB is free.
- `scripts/t1linear_to_2mm.py` resamples the cropped images to 2 mm float32 (85×105×90, linear interpolation): 1.5 GB in total, uploaded to Drive as one `.tar`. At load time (`--data t1linear`): zero-padding to 96×112×96, then z-normalization.

## v2 results (2026-10-29): same experiments on t1-linear images

- Same setup as v1 (DenseNet121, age normalized in the model, lr 1e-4, MSE loss, early stopping on validation MAE, `best-mae` checkpoint), only the input images change. Drive folders: `outputs/site_shift_test-<site>_t1linear`.
- IOP run: stopped at epoch 20 (best epoch 9). The validation curve is again very noisy (MAE 6-21 years from one epoch to the next), so the best-epoch choice is partly luck.

| Held out | insite MAE | outofsite MAE | outofsite bias | outofsite r | v1 outofsite MAE |
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
- In-site MAE is similar to v1 (6.8-7.1 vs 5.8-7.9).
- Caveat: t1-linear combines N4 and affine registration; which of the two removes the Guys / HH shift is not known (an ablation could separate them). Single seed per run.

## v3a: scanner-contrast augmentation

- Same as v1, plus `--augment`: during training only, `RandomScannerContrast` rescales each image to [0, 1], applies a random gamma (log-uniform, 0.70-1.42) and, half of the time, a smooth bias field (coefficients 0.3), then maps the image back to its original intensity range. Applied with probability 0.8. Validation and test images are never augmented (ClinicaDL eval mode, checked).
- Why not re-normalize after the gamma: the cached images were cropped / padded after z-normalization, so re-normalizing shifts intensities (max difference 1.1 z-units even at gamma = 1) and training images would no longer match validation / test images. Mapping back to the original range keeps gamma = 1 an exact identity (difference 2e-7).
- Why this gamma range (calibrated on 12 Guys + 12 HH subjects, bright − intermediate contrast): gamma 0.7 → 1.4 moves the Guys median from 0.81 to 1.48 and HH from 0.68 to 1.18, i.e. about twice the Guys / HH gap (0.94 vs 0.78). IOP (1.81) stays out of reach of a reasonable gamma, so the augmentation may help less for IOP. Caveat: the range is close to TorchIO's default (log-gamma ±0.3), but its magnitude was sanity-checked on Guys and HH images, which are the held-out hospitals in two of the three runs. A strictly clean protocol would calibrate on the training hospitals of each run only.
- Run on the t1-linear images (`--data t1linear --augment`, folders `outputs/site_shift_test-<site>_t1linear_aug`): the geometric part of the shift is already removed by v2, and IOP's remaining problem is contrast, which is what this augmentation targets. (No run of the augmentation on the v1 images.)
- Cost: the bias field is computed on the CPU for every training image (0.20 s vs 0.07 s per image locally), so an epoch takes ~1 min 40 s on Colab, and training runs longer (IOP: 46 epochs, ~80 min).

### v3a results (2026-10-29)

| Held out | insite MAE | outofsite MAE | outofsite bias | outofsite r | v2 outofsite MAE / r |
|---|---|---|---|---|---|
| IOP | 5.66 | **8.80** | +0.93 | **0.82** | 11.21 / 0.52 |
| HH | 7.87 | 9.26 | −6.48 | 0.85 | 8.62 / 0.85 |
| Guys | 7.01 | 10.91 | +9.24 | 0.80 | 8.87 / 0.75 |

Out-of-site signed error by age group (bias in years):

| Age | IOP: n / bias | HH: n / bias | Guys: n / bias |
|---|---|---|---|
| ≤ 35 | 31 / +7.89 | 56 / +3.56 | 68 / +17.56 |
| 35-50 | 20 / +1.66 | 38 / −4.70 | 76 / +12.70 |
| 50-65 | 6 / −6.27 | 57 / −13.30 | 103 / +6.27 |
| 65+ | 11 / −16.09 | 30 / −14.54 | 67 / +1.46 |

- **IOP, the contrast shift, improves clearly**: out-of-site MAE 11.2 → 8.8, correlation 0.52 → 0.82, bias of the over-65s −28 → −16 years. In-site improves as well (6.84 → 5.66, r 0.90): the augmentation also regularizes.
- **HH and Guys, already fixed by v2, do not improve**: MAE +0.6 (HH) and +2.0 (Guys). The ranking is unchanged or better (r 0.85 → 0.85, 0.75 → 0.80); what changes is the global offset, whose sign even flips for HH (+4.8 → −6.5).
- Reading: augmentation helps where the remaining shift is contrast. Elsewhere, the out-of-site MAE is dominated by an offset that moves from one run to the next, consistent with the very noisy validation curves and the best-epoch choice. Differences of 1-2 years between single runs cannot be interpreted; several seeds are needed.
- This offset is also what AdaBN and DANN act on (target statistics, aligned features), which makes them the natural next step.

## v3b: AdaBN (2026-10-30)

- `scripts/adabn.py`, no training: copy of a trained model whose 121 BatchNorm layers have their running statistics reset and re-estimated (cumulative average, batches of 8, no augmentation) on all the images of the held-out hospital; every learned weight is unchanged (checked). The ages of the held-out hospital are never used (the label column passed to ClinicaDL is filled with zeros). Transductive setting: the same images are then predicted. In-site predictions use the original model.
- Applied to the 6 trained models of v2 and v3a; ~1-2 min per model on a T4.

| Run | outofsite MAE before → AdaBN | bias before → AdaBN | r before → AdaBN |
|---|---|---|---|
| IOP (v2) | 11.21 → 14.87 | −4.05 → +10.38 | 0.52 → 0.57 |
| IOP (v3a) | 8.80 → 10.24 | +0.93 → +7.13 | 0.82 → 0.83 |
| HH (v2) | 8.62 → **7.58** | +4.82 → +4.05 | 0.85 → 0.86 |
| HH (v3a) | 9.26 → **7.63** | −6.48 → +1.45 | 0.85 → 0.85 |
| Guys (v2) | 8.87 → 9.13 | +2.51 → −2.42 | 0.75 → 0.74 |
| Guys (v3a) | 10.91 → **8.43** | +9.24 → −3.78 | 0.80 → 0.79 |

- **The correlation never changes** (±0.05): AdaBN only moves the offset of the predictions, as expected from re-estimating normalization statistics.
- **HH and Guys**: AdaBN removes the run-to-run offset. The two HH models, whose biases had opposite signs (+4.8 / −6.5), both land at MAE ≈ 7.6; the two Guys models at 8.4-9.1. Best out-of-site MAE so far for HH (7.58).
- **IOP gets worse** (11.2 → 14.9, 8.8 → 10.2): young IOP subjects become strongly over-estimated (≤ 35: +19 / +13 years).
- **Explanation: label shift.** AdaBN re-centers the features on the target population, so it also removes the part of the feature shift that comes from a *different age distribution*, and predictions are pulled toward the training mean age. The sign and size of the post-AdaBN bias follow the gap between the mean age of the training hospitals and of the target:

| Held out | Mean age, training hospitals | Mean age, target | Gap (train − target) | Bias after AdaBN (v2 / v3a) |
|---|---|---|---|---|
| IOP | 49.5 | 42.4 | +7.1 | +10.4 / +7.1 |
| HH | 49.3 | 47.4 | +1.9 | +4.1 / +1.5 |
| Guys | 46.0 | 50.8 | −4.8 | −2.4 / −3.8 |

(mean ages over all subjects of each hospital; the model's normalization uses train + validation only, which is close.)

- Consequence for DANN: it also aligns the feature distributions of source and target as a whole, so it has the same weakness when the target population is younger or older (IOP). Options: re-weight the source samples to match an estimated target age distribution, a smaller alignment weight, or alignment conditioned on the predicted age. To be stated explicitly when presenting results.

## Stabilized training (2026-10-30)

- `--scheduler plateau`: ReduceLROnPlateau on validation MAE (factor 0.5, patience 3, min 1e-6); early stopping patience 15 instead of 10. Checked locally that the learning rate is halved when the validation MAE stalls. (ClinicaDL's `learning_rates/optimizer.tsv` fills the last epoch's rows with the value after its final reduction.)
- `--seed`: training randomness only (initialization, batch order, augmentation); the data split stays seed 42, so all seeds are tested on the same subjects.
- Augmentation: the bias field is computed on a 4× coarser grid and upsampled (same field within 0.8 %, ~60× faster to compute). The v3a runs used the full-resolution field.
- Augmented runs now take ~30 min (25-29 epochs) instead of ~80.

### Results: IOP, t1-linear + augmentation + scheduler, seeds 42 / 1 / 2

| Seed | Stopped at | Best validation epoch | outofsite MAE, best epoch → final model | insite MAE, best → final | outofsite after AdaBN, final model |
|---|---|---|---|---|---|
| 42 | 29 | 13 (isolated dip) | 11.52 → **8.86** | 6.15 → 7.44 | 10.33 |
| 1 | 25 | 9 (isolated dip) | 13.50 → **11.99** | 6.82 → 6.13 | 11.75 |
| 2 | 50 (limit) | ~40 (converged) | 12.06 → **11.72** | 6.22 → 5.88 | 9.99 |
| **mean ± std** | | | 12.36 ± 1.02 → **10.86 ± 1.73** | 6.40 ± 0.37 → 6.48 ± 0.84 | 10.69 ± 0.93 |

- **The scheduler makes the curves calmer** (after the first learning-rate decreases, validation MAE stays within ~1 year, and seed 2 converges smoothly to ~5.8-6.0), but the first ~15 epochs are still noisy.
- **"Best validation epoch" selection picks lucky dips**: for seeds 42 and 1, the best epoch is an isolated dip in the noisy phase, and early stopping then ends training (patience counted from that dip) while the model is still converging to the same level.
- **The final model is better out-of-site for every seed** (−1.5 years on average), with the same in-site MAE: from now on, every method is evaluated on the final model (also the warm start of DANN).
- **Seed-to-seed spread**: ±0.4-0.8 years in-site, **±1.0-1.7 years out-of-site** on IOP. Differences below ~2 years between single runs are not interpretable.
- **The single v3a IOP run (8.80) was a favorable draw**: this configuration gives 10.9 ± 1.7 on IOP (seed 42's final model: 8.86). The v3a conclusion "augmentation helps IOP" still needs a multi-seed comparison without augmentation (v2 single run: 11.21, r 0.52); the correlation gain (0.52 → 0.66-0.82) looks more robust than the MAE gain.
- AdaBN on the final models: 10.69 ± 0.93, no clear change on IOP (positive bias, label shift).
- Protocol to adopt for the next runs: fixed training length with the scheduler (no early stopping on a noisy curve), final model.

## v3b: DANN (2026-10-30, implemented, to run)

`scripts/train_dann.py`, notebook section 16. Fine-tunes a finished section-15 run (same site, augmentation and seed).

- **Architecture**: the DenseNet features f (1024 values after global pooling) feed the original age head and a domain classifier (1024 → 256 → 256 → 1, ReLU, dropout 0.5) through a gradient reversal layer (identity forward, gradient × −λ backward). Domain label: 0 = training hospitals (source), 1 = held-out hospital (target).
- **Warm start** from the source model: fine-tuning a model that already predicts age is much more stable than training both objectives from scratch on 300-400 images, and gives a paired "before / after" comparison. A control with λ = 0 (same fine-tuning, no domain loss) separates the DANN effect from the effect of 30 more epochs.
- **Loss scales**: the age loss is the MSE on normalized ages (z units, ~0.1-0.5), the domain loss a binary cross-entropy (~0.69 at chance level). With the age MSE in years² (~25-100), λ would be meaningless (the domain term would be ~100× too weak).
- **BatchNorm statistics frozen** (the source statistics of the warm-start model), affine BatchNorm parameters still trained: the alignment must be learned by the weights; training and test use the same normalization; 8 + 8 images per batch would make batch statistics noisy; and changing the statistics is AdaBN's role, kept as a separate method.
- **Schedules from the original paper** (Ganin et al., JMLR 2016): λ_p = λ_max (2 / (1 + e^(−10p)) − 1), rising from 0 so that the domain classifier is trained before its reversed gradient matters; learning rate lr_p = lr / (1 + 10p)^0.75. p goes from 0 to 1 over the training steps. Learning rates: 3e-5 for the pretrained network (fine-tuning), 1e-4 for the new domain classifier; Adam; 30 epochs; batches of 8 source (augmented as in the source run) + 8 target images (never augmented; the target set is cycled).
- **IW-DANN against label shift** (`--iw`, Tachet des Combes et al., NeurIPS 2020): source images are re-weighted in the domain loss by w(age bin) = p_target(bin) / p_source(bin), with 5 bins at the source age quintiles. p_target is re-estimated each epoch from the model's *predicted* target ages (+1 smoothing; weights clipped to [0.25, 4]); true target ages are never used. Aligning the re-weighted source with the target no longer asks the features to erase a real age difference, which is what made AdaBN fail on IOP. Limitation: the predicted target ages are themselves biased by the shift, so the estimate is approximate.
- **Model selection without target labels**: the main result is the final model (fixed training length, as in the paper); the best source-validation checkpoint is saved as a check that age prediction on the source does not degrade. Target ages must not be used to choose λ: all configurations are reported (sensitivity), and the main configuration is fixed in advance (IW-DANN, λ 0.3).
- **Monitoring** (`dann_log.tsv`, `dann_curves.png`): source validation MAE, domain classifier accuracy (0.5 = domains indistinguishable; staying near 1 means no alignment, a collapse of the source MAE means too strong a λ), mean predicted target age (unlabeled statistic).
- Transductive setting, as for AdaBN: the target images used for adaptation (without ages) are the evaluated images.
- Checked locally: the DANN age output equals the warm-start model's; the GRL gradient is −λ; IW weights up-weight the source bins where the predicted target ages fall; end-to-end run on a small set; outputs readable by `predict_ages.py` / `adabn.py` (`--run-dir ... --checkpoint final`).

### DANN results: IOP held out, seed 42 (2026-10-30)

All runs start from the final model of `IOP_t1linear_aug_plateau` (seed 42); 316 source training images, 80 source validation, 68 target images; ~76 s per epoch on a T4 (~38 min per configuration).

| Method | outofsite MAE | bias | ≤ 35 bias | 65+ bias | r | domain acc (end) | mean predicted IOP age (end) | source val MAE (end) |
|---|---|---|---|---|---|---|---|---|
| source model (final) | 8.86 | −1.44 | | | | | 40.9 | 6.30 |
| + AdaBN | 10.33 | +5.48 | | | | | | |
| control (λ = 0) | **8.87** | −4.21 | +2.82 | −21.87 | 0.77 | 0.96 | 38.2 | 5.62 |
| DANN λ = 0.1 | 11.57 | +6.93 | +13.61 | −10.15 | 0.74 | 0.53 | 49.3 | 5.65 |
| DANN λ = 0.3 | 13.03 | +9.77 | +16.38 | −7.51 | 0.73 | 0.48 | 52.2 | 5.69 |
| IW-DANN λ = 0.3 | 10.48 | +4.49 | +11.49 | −13.00 | 0.76 | 0.52 | 46.8 | 5.63 |

(True mean age: IOP 42.4, training hospitals 49.5; given for interpretation only, never used by the methods.)

- **The features encode the hospital**: in the control (no reversed gradient), the domain classifier reaches 96 % accuracy on the 1024 DenseNet features.
- **Extra fine-tuning alone** (control) improves the source (validation MAE 6.30 → 5.62) but not IOP (8.86 → 8.87): predictions get younger (bias −1.4 → −4.2), better for young IOP subjects and worse for the over-65s (−22 years). The training loss falls to 0.01 (z units, ≈ 1.6 years): the model memorizes its 316 images. Between epochs the mean predicted IOP age moves between 33 and 42 years while the source validation MAE stays within 5.5-5.9: predictions on the unseen hospital are much less stable than on the source.
- **DANN aligns the domains as intended** (domain accuracy ≈ 0.5 from epoch ~8, source validation MAE unchanged, IOP ranking kept, r ≈ 0.73-0.74) **and makes IOP worse, with a dose-response**: λ 0 → 0.1 → 0.3 gives a mean predicted IOP age of 38 → 49 → 52 years and a bias of −4 → +7 → +10. Making IOP's features indistinguishable from the source also makes IOP's patients look as old as (or older than) the source's: label shift, the same signature as AdaBN (young subjects over-estimated) with a different method.
- **IW-DANN corrects about half of it**: at the same λ, the bias drops from +9.8 to +4.5, the MAE from 13.0 to 10.5, and the mean predicted IOP age plateaus around 44-46 instead of 52. It is still worse than the control (gap 1.6 years, within the seed-to-seed noise of ±1.7, so not conclusive on one seed). A likely reason for the partial correction: the target age distribution is estimated from predicted ages, which the alignment itself pushes up, so the weights correct less and less (feedback loop; to check in the `w0`-`w4` columns of `dann_log.tsv`).
- **Conclusion for IOP**: no unsupervised adaptation method beats not adapting. IOP differs both in scanner (contrast) and in population (7 years younger); from images alone, aligning the scanner difference also erases part of the age difference. Robust on one seed: the 4-year degradation of DANN λ 0.3 with its dose-response, and the halving of the bias by IW. Not robust: IW-DANN vs control.
- Next run (notebook `site_dann_pipeline.ipynb`): the same on **HH**, whose population is close to the training hospitals' (gap 1.9 years vs 7.1 for IOP) and where AdaBN helped. Prediction: DANN should help, or at least not hurt, on HH.

## Interpretation so far

- **The model learns on its training hospitals**: in-site MAE 5.8-7.9 years vs ≈ 14 for the naive baseline, with the usual mild regression to the mean.
- **On an unseen hospital the error is large and systematic, and its direction depends on the hospital**: HH is under-estimated (−25 years on average), Guys over-estimated (+17), IOP under-estimated only for older subjects. "An unseen hospital makes brains look younger" is therefore wrong.
- **The ranking of subjects survives the shift** (r ≈ 0.7-0.8 on every unseen hospital): what breaks is a site-specific offset and slope (predicted ≈ a + b × true age, with a and b depending on the hospital).
- Hypothesis: some scanner-dependent image property is partly read by the model as age, with HH and Guys on opposite sides of it. The image statistics rule out field of view / background normalization for Guys vs HH and show clear per-site contrast fingerprints, but no global intensity measure explains the direction of the bias.
- The hypothesis "IOP is younger, so the gap is a population effect" is contradicted by the IOP run: the IOP gap comes from its older subjects.
- v1 improves both IOP MAEs over v1a and stabilizes training, but there is a single run each; differences of a few years may be chance.
- A train/validation gap appears in v1 (≈ 2.5 vs ≈ 6-7 years RMSE / MAE), i.e. some overfitting.
- **Update after t1-linear**: the Guys / HH shift was mostly geometric or bias-field related (removed by N4 + affine registration), not a global intensity effect. Two kinds of shift remain distinct: preprocessing fixes HH and Guys, while IOP's shift (contrast) survives it.
- **Update after v3a / v3b**: contrast augmentation fixes much of IOP's shift; the remaining HH / Guys error is mostly an offset, which AdaBN removes. AdaBN fails on IOP because of label shift (IOP is 7 years younger than the training hospitals on average): a scanner shift and a population shift cannot be told apart from images alone.
- **Update after multi-seed runs and DANN**: single runs vary by ±1.7 years out-of-site, so v3a's IOP gain was partly luck (10.9 ± 1.7 over 3 seeds). DANN on IOP confirms the label-shift explanation with a dose-response (stronger alignment → older predictions for IOP), and IW-DANN halves the effect without removing it. The working hypothesis is now: alignment-based adaptation helps when the target population matches the source (HH, Guys with AdaBN) and hurts under label shift (IOP).

**Limitations**: one run per configuration; training sets of different sizes (249-495 subjects); small age groups in some test sets; no skull stripping; very noisy validation curves (the best-epoch choice is partly luck).

## Next ideas

After v2, v3a and AdaBN, in order:

- **Stabilize training** before comparing methods: learning-rate scheduler, possibly larger batches or GroupNorm instead of BatchNorm, and several seeds per configuration.
- **DANN on HH** (running), then possibly Guys: does alignment help when there is little label shift?
- **More seeds** for the control and IW-DANN on IOP (the 1.6-year gap is within the noise).
- **Better target age estimate for IW-DANN**: estimate the target age distribution once from the source model (before alignment) and keep it fixed, to break the feedback loop; or use black-box shift estimation (confusion-matrix correction).
- **AdaBN without transduction**: re-estimate the statistics on half of the target images and evaluate on the other half.
- **Faster augmentation**: compute the bias field on a coarse grid and upsample it.
- **Ablation N4 vs registration**, to know which step removes the Guys / HH shift.

Earlier ideas (before t1-linear):

- **Signed predictions for v1a** (notebook cell 7 bis) to check whether v1a shows the same pattern.
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
