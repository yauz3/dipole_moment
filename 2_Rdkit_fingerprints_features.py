#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 22/12/2025
# Author: Sadettin Y. Ugurlu



import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import rdFingerprintGenerator

# İsterseniz RDKit uyarılarını kapatın (opsiyonel)
# RDLogger.DisableLog("rdApp.error")
# RDLogger.DisableLog("rdApp.warning")

def fp_to_np(fp, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr

def empty_block(n: int) -> np.ndarray:
    return np.zeros((n,), dtype=np.int8)

def get_morgan_fp(mol, radius: int, n_bits: int, use_features: bool) :
    """
    RDKit sürüm uyumlu Morgan FP üretimi.
    - ECFP: use_features=False -> rdFingerprintGenerator.GetMorganGenerator(...)
    - FCFP: use_features=True  -> mümkünse MorganFeatureAtomInvGen, yoksa AllChem fallback
    """
    if mol is None:
        return None

    if not use_features:
        # ECFP
        try:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
            return gen.GetFingerprint(mol)
        except Exception:
            # Fallback (eski API)
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=int(radius), nBits=int(n_bits), useFeatures=False)

    # FCFP
    try:
        # Bazı RDKit sürümlerinde bu fonksiyon vardır
        if hasattr(rdFingerprintGenerator, "GetMorganFeatureAtomInvGen"):
            atom_inv_gen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=int(radius),
                fpSize=int(n_bits),
                atomInvariantsGenerator=atom_inv_gen
            )
            return gen.GetFingerprint(mol)
    except Exception:
        pass

    # Son çare: AllChem (useFeatures=True)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=int(radius), nBits=int(n_bits), useFeatures=True)

def smiles_to_many_fingerprints(
    smiles_list,
    bits_list=(128, 256, 512, 1024),
    levels=(1, 2, 3),              # "radius-benzeri" seviye
    include_avalon=True,
    include_maccs=True,
    include_ecfp=True,
    include_fcfp=True,
    include_rdk=True,
):
    smiles_list = list(smiles_list)

    # mol'ları bir kez üret
    mols = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol = None
        mols.append(mol)

    all_blocks = []

    # 36 blok = (ECFP + FCFP + RDK) * (4 bit) * (3 seviye)
    for nBits in bits_list:
        for lv in levels:
            r = int(lv)

            # 1) ECFP (Morgan, useFeatures=False)
            if include_ecfp:
                block = []
                for mol in mols:
                    if mol is None:
                        block.append(empty_block(nBits))
                    else:
                        fp = get_morgan_fp(mol, radius=r, n_bits=nBits, use_features=False)
                        block.append(fp_to_np(fp, nBits))
                df = pd.DataFrame(block, columns=[f"ECFP_r{r}_b{nBits}_{i}" for i in range(nBits)])
                all_blocks.append(df)

            # 2) FCFP (Morgan, useFeatures=True)
            if include_fcfp:
                block = []
                for mol in mols:
                    if mol is None:
                        block.append(empty_block(nBits))
                    else:
                        fp = get_morgan_fp(mol, radius=r, n_bits=nBits, use_features=True)
                        block.append(fp_to_np(fp, nBits))
                df = pd.DataFrame(block, columns=[f"FCFP_r{r}_b{nBits}_{i}" for i in range(nBits)])
                all_blocks.append(df)

            # 3) RDK path FP: level -> maxPath (3,5,7)
            if include_rdk:
                maxPath = 2 * r + 1
                block = []
                for mol in mols:
                    if mol is None:
                        block.append(empty_block(nBits))
                    else:
                        fp = Chem.RDKFingerprint(
                            mol,
                            fpSize=int(nBits),
                            minPath=1,
                            maxPath=int(maxPath),
                            nBitsPerHash=2
                        )
                        block.append(fp_to_np(fp, nBits))
                df = pd.DataFrame(block, columns=[f"RDK_p{maxPath}_b{nBits}_{i}" for i in range(nBits)])
                all_blocks.append(df)

    # Ek bloklar: Avalon (4), MACCS (1)
    if include_avalon:
        for nBits in bits_list:
            block = []
            for mol in mols:
                if mol is None:
                    block.append(empty_block(nBits))
                else:
                    fp = GetAvalonFP(mol, nBits=int(nBits))
                    block.append(fp_to_np(fp, int(nBits)))
            df = pd.DataFrame(block, columns=[f"Avalon_b{nBits}_{i}" for i in range(nBits)])
            all_blocks.append(df)

    if include_maccs:
        maccs_bits = 167
        block = []
        for mol in mols:
            if mol is None:
                block.append(empty_block(maccs_bits))
            else:
                fp = MACCSkeys.GenMACCSKeys(mol)
                block.append(fp_to_np(fp, maccs_bits))
        df = pd.DataFrame(block, columns=[f"MACCS_{i}" for i in range(maccs_bits)])
        all_blocks.append(df)

    return pd.concat(all_blocks, axis=1)

# ========= Run =========
train_df = pd.read_csv("dipole_moments_with_smiles.csv")
train_smiles = train_df["SMILES"].astype(str).tolist()

fps_df = smiles_to_many_fingerprints(
    train_smiles,
    bits_list=(128, 256, 512, 1024),
    levels=(0, 1, 2, 3),
    include_avalon=True,   # +4 blok
    include_maccs=True,    # +1 blok
    include_ecfp=False,     # 12 blok
    include_fcfp=False,     # 12 blok
    include_rdk=False,      # 12 blok
)

final_df = pd.concat([train_df.reset_index(drop=True), fps_df.reset_index(drop=True)], axis=1)
final_df.to_csv("dipole_fps_36plus.csv", index=False)

print("OK. Output: dipole_fps_36plus.csv")
print("Total FP columns:", fps_df.shape[1])
