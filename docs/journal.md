# Journal d'expériences

Notes de travail : configurations, résultats bruts, interprétations provisoires et pistes.
Les résultats sont préliminaires (un seul entraînement par configuration).

## Run v1 (28/10/2026) : référence

**Configuration**

- Entraînement sur Guys + HH : 495 sujets, dont 99 mis de côté pour le test interne (`insite`) ; 1er fold d'un 5-fold sur le reste (train / validation).
- Test sur IOP, hôpital jamais vu (`outofsite`) : 68 sujets.
- Prétraitement : ToCanonical, rééchantillonnage 2 mm, ZNormalization, CropOrPad 112 × 128 × 112. **Pas de recalage ni de retrait du crâne.**
- DenseNet121 3D (MONAI), sortie = âge brut en années, loss MSE, Adam lr 1e-3, batch 8, AMP.
- Early stopping (patience 10) sur la MAE de validation ; évaluation avec le checkpoint `best-mae`.
- Colab GPU T4 : ~30 s par epoch (après pré-calcul du prétraitement). Arrêt à l'epoch 38, meilleure epoch : 27.
- Dossier sur Drive : `outputs/site_shift_test-IOP_v1_lr1e-3` (après archivage).

**Résultats**

| | Modèle | Référence naïve (âge moyen Guys + HH = 49,5 ans) |
|---|---|---|
| insite (Guys + HH) | **6,16 ans** | 14,2 ans* |
| outofsite (IOP) | **12,21 ans** | 16,2 ans |

\* calculée sur tous les sujets Guys + HH, pas seulement sur les 99 du test interne.

**MAE par tranche d'âge**

| Âge | insite | outofsite (IOP) |
|---|---|---|
| 20-35 | 4,73 (n=28) | 4,85 (n=30) |
| 35-50 | 8,49 (n=28) | 12,41 (n=20) |
| 50-65 | 6,30 (n=27) | 21,97 (n=6) |
| 65-90 | 4,37 (n=16) | 27,19 (n=11) |

Un sujet IOP de 20,0 ans manque (borne gauche exclue par `pd.cut`).

**Populations**

| | Guys | HH | IOP |
|---|---|---|---|
| Sujets | 314 | 181 | 68 |
| Âge moyen / médian | 50,8 / 53,4 | 47,4 / 48,0 | 42,4 / 36,3 |
| F / M | 175 / 139 | 94 / 87 | 44 / 24 |

**Courbe de validation** : très instable, la MAE oscille entre ~7 et ~52 ans d'une epoch à l'autre. Le choix du meilleur checkpoint profite donc en partie du hasard.

**Interprétation provisoire**

- Sur ses hôpitaux d'entraînement, le modèle apprend réellement (erreur divisée par 2,3 par rapport à la référence naïve).
- Sur IOP, il ne fait qu'un peu mieux que la référence naïve.
- La dégradation n'est **pas uniforme** : nulle chez les 20-35 ans, très forte après 50 ans. L'hypothèse « IOP est plus jeune, donc l'écart vient de la population » est contredite : les jeunes d'IOP sont aussi bien prédits que les autres.
- Hypothèse (non vérifiée) : les marqueurs du vieillissement appris par le modèle (atrophie, contraste substance grise / blanche, ventricules) n'apparaissent pas de la même façon sur le scanner d'IOP.

**Limites** : un seul run, entraînement instable, tranches âgées d'IOP très petites (6 et 11 sujets), erreur absolue seulement (le sens de l'erreur est inconnu, d'où `predict_ages.py`).

## Run v2 (à lancer) : stabilisation

- L'âge est normalisé à l'intérieur du modèle (`AgeRegressor` : sortie = moyenne + écart-type × z, statistiques calculées sur train + validation), learning rate 1e-4.
- Labels, loss et MAE restent en années : les résultats sont directement comparables à la v1.
- Questions : la courbe de validation est-elle plus stable ? Le schéma par tranche d'âge sur IOP se maintient-il ?

## Pistes à garder

- **Prédictions signées** (`scripts/predict_ages.py`) : le modèle sous-estime-t-il l'âge des sujets âgés d'IOP ? Le faire sur v1 et v2.
- **Inverser les rôles** : `--test-site Guys` et `--test-site HH`, pour savoir si l'effet est propre à IOP.
- **Plusieurs seeds** : les seeds sont fixés à 42 (split et entraînement) ; il faudra une option `--seed` pour mesurer la variabilité.
- **Correction du biais vers la moyenne**, classique en âge cérébral (régression linéaire de l'erreur sur l'âge, ajustée sur la validation).
- **Robustesse aux scanners** : augmentations d'intensité / de contraste (dans `IMAGE_TRANSFORMS`), harmonisation entre sites.
- **Prétraitement plus complet** : recalage (par exemple Clinica `t1-linear`) et retrait du crâne.
- **Pratique** : garder `bids_2mm` sur Drive pour éviter le pré-calcul à chaque session ; si l'apprentissage est plus lent avec lr 1e-4, augmenter `--epochs` et la patience.

## Pièges techniques rencontrés (ClinicaDL 2.0)

- Noms de groupe de test sans tiret ni underscore (`insite`, pas `in-site`) : un tiret casse la relecture du MAPS.
- `OptimizationConfig(num_epochs=...)` ; `make_split` produit `train.tsv` ; `KFold` prend le dossier renvoyé par `make_kfold` ; les epochs de `ModelCheckpointCallback` commencent à 1.
- Early stopping : la métrique surveillée doit être une métrique calculée (`mae`, pas `loss`) ; `MAEMetricConfig(label_key="AgeMRI")`.
- `trainer.test()` suppose un GPU par défaut : lui passer le même `ComputationalConfig`.
- Sur CPU, `amp` et `channels_last` ralentissent énormément les convolutions 3D.
- ClinicaDL refuse d'écraser un MAPS ou un split existant (d'où `clean_run()` dans le notebook).
- Le test n'enregistre que l'erreur absolue par sujet (`details.tsv`), pas les prédictions.
- Checkpoints : `maps/training/split-0/models/best_models/best-mae/model.pt`, clés préfixées par `network.`.
- `git pull` dans Colab met à jour les scripts, pas le notebook ouvert : rouvrir le notebook depuis GitHub.
- Pré-calcul du prétraitement : chargement d'une image 0,53 s → 0,08 s (mesuré sur le PC).
