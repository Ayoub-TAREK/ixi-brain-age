"""
Convertit les IRM T1 d'IXI en BIDS + crée le fichier des âges.

Entrée  : data/IXI/raw/IXI002-Guys-0828-T1.nii.gz ...   (extraits de IXI-T1.tar)
          data/IXI/IXI.xls                               (démographie)
Sortie  : data/IXI/bids/
            dataset_description.json        (DatasetType = raw)
            sub-IXI002/ses-M000/anat/sub-IXI002_ses-M000_T1w.nii.gz
            participants_sessions_age.tsv   (participant_id, session_id, AgeMRI, sex, site)

Dépendances : pip install pandas xlrd
Lancer depuis la racine du projet :  python scripts/ixi_to_bids.py
"""

import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IXI_DIR = BASE_DIR / "data" / "IXI"
RAW_DIR = IXI_DIR / "raw"
XLS_PATH = IXI_DIR / "IXI.xls"
BIDS_DIR = IXI_DIR / "bids"

# Nom de fichier IXI : IXI<ID>-<HOPITAL>-<n°>-T1.nii.gz   (HOPITAL = Guys, HH ou IOP)
PATTERN = re.compile(r"IXI(\d{3})-(Guys|HH|IOP)-\d+-T1\.nii\.gz$")
SEX_MAP = {1: "M", 2: "F"}  # codage IXI : 1 = homme, 2 = femme


def link_or_copy(src: Path, dst: Path) -> None:
    """Lien physique (pas d'espace disque en plus) ; copie si impossible."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_demographics() -> pd.DataFrame:
    demo = pd.read_excel(XLS_PATH)
    demo = demo[["IXI_ID", "SEX_ID (1=m, 2=f)" if "SEX_ID (1=m, 2=f)" in demo.columns else "SEX_ID", "AGE"]]
    demo.columns = ["IXI_ID", "SEX_ID", "AGE"]
    demo = demo.dropna(subset=["AGE"])
    # certains IDs apparaissent plusieurs fois : on garde le premier âge renseigné
    demo = demo.drop_duplicates(subset="IXI_ID", keep="first")
    return demo.set_index("IXI_ID")


def main():
    demo = load_demographics()
    print(f"Sujets avec un âge dans IXI.xls : {len(demo)}")

    BIDS_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing_age, unmatched = [], 0, 0

    for nii in sorted(RAW_DIR.glob("*.nii.gz")):
        m = PATTERN.search(nii.name)
        if not m:
            unmatched += 1
            continue
        ixi_id, site = int(m.group(1)), m.group(2)
        if ixi_id not in demo.index:
            missing_age += 1
            continue

        sub = f"sub-IXI{ixi_id:03d}"
        ses = "ses-M000"
        anat = BIDS_DIR / sub / ses / "anat"
        anat.mkdir(parents=True, exist_ok=True)
        link_or_copy(nii, anat / f"{sub}_{ses}_T1w.nii.gz")

        sex = demo.loc[ixi_id, "SEX_ID"]
        rows.append({
            "participant_id": sub,
            "session_id": ses,
            "AgeMRI": round(float(demo.loc[ixi_id, "AGE"]), 2),
            "sex": SEX_MAP.get(int(sex), "n/a") if pd.notna(sex) else "n/a",
            "site": site,
        })

    with open(BIDS_DIR / "dataset_description.json", "w") as f:
        json.dump({
            "Name": "IXI T1 (converted for ClinicaDL)",
            "BIDSVersion": "1.9.0",
            "DatasetType": "raw",
            "License": "CC BY-SA 3.0",
            "HowToAcknowledge": "IXI dataset, https://brain-development.org/ixi-dataset/",
        }, f, indent=2)

    table = pd.DataFrame(rows)
    table.to_csv(BIDS_DIR / "participants_sessions_age.tsv", sep="\t", index=False)

    print(f"Images converties : {len(table)}")
    print(f"Images sans âge (ignorées) : {missing_age}")
    print(f"Fichiers au nom inattendu (ignorés) : {unmatched}")
    print("\nPar hôpital :")
    print(table.groupby("site")["AgeMRI"].agg(["count", "mean", "min", "max"]).round(1))


if __name__ == "__main__":
    main()