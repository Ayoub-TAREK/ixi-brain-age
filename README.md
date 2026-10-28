# ixi-brain-age

Prédiction de l'âge cérébral à partir d'IRM T1 du jeu de données [IXI](https://brain-development.org/ixi-dataset/), avec [ClinicaDL 2.0](https://clinicadl.readthedocs.io/) et un DenseNet 3D (MONAI).

L'objectif est de mesurer l'effet d'un **changement de domaine entre hôpitaux** (site shift) :

- entraînement sur les hôpitaux **Guys** et **HH** (split par patient : train / validation / test interne) ;
- évaluation sur le test interne (mêmes hôpitaux, sujets jamais vus) ;
- évaluation sur l'hôpital **IOP**, jamais vu pendant l'entraînement.

## Contrôle du prétraitement

Le prétraitement est pré-calculé une seule fois par `scripts/cache_preprocessed.py` (images à 2 mm dans `data/IXI/bids_2mm`), ce qui accélère l'entraînement.

Une image par hôpital après prétraitement (sortie de `--check`) :

| Guys | HH | IOP |
|:---:|:---:|:---:|
| ![Guys](docs/check_Guys_sub-IXI002.png) | ![HH](docs/check_HH_sub-IXI012.png) | ![IOP](docs/check_IOP_sub-IXI035.png) |

## Structure

```
scripts/ixi_to_bids.py                conversion IXI -> BIDS + fichier des âges
scripts/train_ixi_site_shift.py       entraînement et évaluation in-site / out-of-site
notebooks/ixi_brain_age_colab.ipynb   exécution sur Google Colab
docs/                                 images de contrôle
```

## Reproduire

Les données IXI **ne sont pas incluses** dans ce dépôt. Elles sont à télécharger depuis la source officielle : https://brain-development.org/ixi-dataset/ (voir les conditions d'utilisation sur ce site).

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Télécharger `IXI-T1.tar` et `IXI.xls`, puis les placer ainsi :
   ```
   data/IXI/raw/IXI002-Guys-0828-T1.nii.gz ...   (contenu extrait de IXI-T1.tar)
   data/IXI/IXI.xls
   ```
3. Convertir en BIDS (sortie dans `data/IXI/bids/`) :
   ```bash
   python scripts/ixi_to_bids.py
   ```
4. Vérifier le prétraitement sur une image par hôpital :
   ```bash
   python scripts/train_ixi_site_shift.py --check
   ```
5. Lancer l'entraînement (options : `--epochs`, `--batch-size`) :
   ```bash
   python scripts/train_ixi_site_shift.py --test-site IOP
   ```

Les résultats sont écrits dans `outputs/site_shift_test-IOP/`.

## Résultats

À venir.
