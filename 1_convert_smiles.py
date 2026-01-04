#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/12/2025
# Author: Sadettin Y. Ugurlu


import pandas as pd
import requests
import time
from urllib.parse import quote
import pubchempy as pcp

# ============================
#  Settings
# ============================

BASE_SLEEP = 0.5          # base sleep time (sec)
MAX_RETRIES = 5           # maximum retries for PubChem
CACTUS_URL = "https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"


# ============================
#  Helper functions
# ============================

def normalize_name(raw_name: str) -> str:
    """
    Light normalization for chemical names to improve PubChem/Cactus resolution.
    Examples:
      '(+)-a-pinene' -> 'alpha-pinene'
      'α-pinene'     -> 'alpha-pinene'
    """
    if not isinstance(raw_name, str):
        return raw_name

    name = raw_name.strip()

    # Remove leading stereochemical signs like (+)- / (-)-
    for prefix in ["(+)-", "(-)-", "(+)", "(-)"]:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()

    # Convert Greek letters to text
    replacements = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)

    # Convert spellings like 'a-pinene' to 'alpha-pinene'
    name = name.replace("a-pinene", "alpha-pinene")

    return name


# ============================
#  Resolver 1: PubChem (pubchempy) + retry/backoff
# ============================

def name_to_smiles_pubchem(name: str) -> str | None:
    """
    Fetch Canonical SMILES from a chemical name using PubChem (pubchempy).
    Retries with exponential backoff for 503/ServerBusy responses.
    """
    if not isinstance(name, str) or not name.strip():
        return None

    norm_name = normalize_name(name)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            compounds = pcp.get_compounds(norm_name, 'name')
            if not compounds:
                print(f"[PubChem] Not found: {name!r} (normalized: {norm_name!r})")
                return None

            smiles = compounds[0].canonical_smiles
            print(f"[PubChem] {name!r} -> {smiles}")
            # If successful, lightly sleep and return
            time.sleep(BASE_SLEEP)
            return smiles

        except Exception as e:
            msg = str(e)
            # Retry for 503 / ServerBusy
            if "ServerBusy" in msg or "503" in msg:
                wait = BASE_SLEEP * (2 ** (attempt - 1))  # 0.5, 1, 2, 4, 8, ...
                print(f"[PubChem] ServerBusy/503 ({name!r}), attempt {attempt}/{MAX_RETRIES}, sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            else:
                # Other errors: do not retry
                print(f"[PubChem] Error ({name!r}): {e}")
                return None

    print(f"[PubChem] Max retries exceeded, giving up: {name!r}")
    return None


# ============================
#  Resolver 2: NCI Cactus
# ============================

def name_to_smiles_cactus(name: str) -> str | None:
    """
    Fetch SMILES from a chemical name using the NCI Cactus resolver.
    """
    if not isinstance(name, str) or not name.strip():
        return None

    norm_name = normalize_name(name)
    encoded_name = quote(norm_name)
    url = CACTUS_URL.format(name=encoded_name)

    try:
        r = requests.get(url, timeout=10)
        print(f"[Cactus] Request: {url} -> status {r.status_code}")
        if r.status_code != 200:
            return None

        smiles = r.text.strip()
        if not smiles or "Not Found" in smiles:
            print(f"[Cactus] Not found: {name!r} (normalized: {norm_name!r})")
            return None

        time.sleep(BASE_SLEEP)
        print(f"[Cactus] {name!r} -> {smiles}")
        return smiles
    except Exception as e:
        print(f"[Cactus] Error ({name!r}): {e}")
        return None


# ============================
#  Main resolver: PubChem first, then Cactus
# ============================

def name_to_smiles(name: str) -> str | None:
    """
    Try PubChem (with retry/backoff) first; if it fails, try NCI Cactus.
    """
    smiles = name_to_smiles_pubchem(name)
    if smiles:
        return smiles

    smiles = name_to_smiles_cactus(name)
    if smiles:
        return smiles

    print(f"[WARN] SMILES not found: {name!r}")
    return None


# ============================
#  CSV processing
# ============================

def add_smiles_to_csv(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)

    if "Name" not in df.columns:
        raise ValueError("'Name' column not found in the CSV. Please check column names.")

    smiles_list = []
    total = len(df)

    for idx, row in df.iterrows():
        name = row["Name"]
        print(f"\n[{idx + 1}/{total}] Processing: {name!r}")
        smiles = name_to_smiles(name)
        smiles_list.append(smiles)

    df["SMILES"] = smiles_list
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV with SMILES column: {output_csv}")


# ============================
#  Run
# ============================

if __name__ == "__main__":
    input_csv_path = "dipole_moments.csv"
    output_csv_path = "dielectric_nd_with_smiles.csv"
    add_smiles_to_csv(input_csv_path, output_csv_path)

